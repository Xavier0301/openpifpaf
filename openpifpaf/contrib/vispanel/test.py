import io
import numpy as np
import openpifpaf
import PIL
import requests
import torch

import json 
import os
import sys
import copy
import glob

import cv2 as cv

from openpifpaf import datasets, decoder, logger, network, plugins, metric, show, transforms, visualizer, annotation,__version__

from torchvision import transforms as torchTransforms

from pycocotools import coco

# from vispanel import VisPanel
# from datamodule import VisPanelModule
from . import vispanel
from . import datamodule

device = torch.device('cpu')

cp_image_dir = "./physical_cp_dataset"
coco_image_dir = "./coco_dataset"
annotations_file = "./physical_cp_dataset/annotations.json"
out_image_dir = "./out"
out_image_dir_gt = "./out_gt"
tmp_image_dir = "./tmp"

wild_images_dir = "./wild_images"

gt_tmp_dir = "./gt_tmp/"

# p = pycocotools.cocoeval.Params('bbox')

# decoder.utils.CifSeeds = 0.3 # u idiot

class FromDir(torch.utils.data.Dataset):
    def __init__(self, directory, n_images=None, preprocess=None):
        self.directory = directory
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.n_images = n_images

        with open(os.path.join(directory, "annotations.json"), 'r') as json_raw:
            json_dict = json.load(json_raw)
            self.annotations = json_dict['annotations']

            self.image_names = [None] * len(json_dict['images'])
            for im_dict in json_dict['images']:
                self.image_names[im_dict['id']-1] = im_dict['file_name'] 

    def __getitem__(self, index):
        image_path = os.path.join(self.directory, self.image_names[index])
        with open(image_path, 'rb') as f:
            image = PIL.Image.open(f).convert('RGB')

        anns = [ann for ann in self.annotations if ann['image_id'] == index+1]
        meta = {
            'dataset_index': index,
            'image_id': index,
            'file_dir': image_path,
            'file_name': self.image_names[index]
        }
        image, anns, meta = self.preprocess(image, anns, meta)
        return image, anns, meta

    def __len__(self):
        if self.n_images is not None:
            return min(self.n_images, len(self.image_names))
        else:
            return len(self.image_names)

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def get_processor():
    model_name = 'vispanel-noise-epoch80.pkl'
    model_path = '/home/xplore/openpifpaf/openpifpaf/contrib/vispanel/models/' + model_name
    print(model_path)
    model_cpu, _ = openpifpaf.network.factory(checkpoint=model_path)
    model = model_cpu.to(device)

    head_metas = [hn.meta for hn in model.head_nets]
    processor = decoder.factory(
        head_metas, profile=None, profile_device=device)
    return processor, model

def get_preprocess():
    rescale_t = transforms.RescaleAbsolute(513)

    pad_t = transforms.CenterPadTight(16)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        rescale_t,
        pad_t,
        transforms.EVAL_TRANSFORM,
    ])

def custom_data_loader(n_images, preprocess):
    train_data = vispanel.VisPanel(
        cp_image_dir=cp_image_dir,
        coco_image_dir=coco_image_dir,
        annotation_file=annotations_file,
        preprocess=preprocess,
        n_images=n_images,
        prediction=True,
        tmpDir=tmp_image_dir,
        randomSeed=512
    )

    return torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False,
        pin_memory=False, collate_fn=datasets.collate_images_anns_meta)

def generic_data_loader(dir, n_images, preprocess):
    data = FromDir(dir, n_images=n_images, preprocess=preprocess)

    return torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False,
        pin_memory=False, collate_fn=datasets.collate_images_anns_meta)

def dumpAnnotations(out_path, image_id, image_name, anns):
    with open("./physical_cp_dataset/annotations.json", 'r') as json_raw:
        annotations_dict = json.load(json_raw)

    annotations_dict['images'] = [{
        'id': image_id,
        'file_name': image_name
    }]
    annotations_dict['annotations'] = anns 

    with open(out_path, 'w') as f:
        json.dump(annotations_dict, f)

def collect(data_loader, file_name_indicator, open_image_dir, paint=False):
    coco_stats = []

    processor, model = get_processor()
    preprocess = get_preprocess()

    # cifdet_seed

    # visualizers
    annotation_painter = show.AnnotationPainter()

    for batch_i, (image_tensors_batch, gts_batch, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=device)

        # unbatch
        for pred, ground_truths, meta, image in zip(pred_batch, gts_batch, meta_batch, image_tensors_batch):
            pred = preprocess.annotations_inverse(pred, meta) #array of annotations

            print("preds:")
            print(pred)
            print(len(ground_truths))

            print(meta['dataset_index'])
            print(meta['file_name'])

            for i, ann in enumerate(ground_truths):
                ann['image_id'] = meta['dataset_index']
                ann['bbox'] = ann['bbox_original'].tolist()
                ann['keypoints'] = ann['keypoints'].tolist()
                ann['id'] = i
                del ann['bbox_original']

            gt_path = gt_tmp_dir + file_name_indicator + "_" + str(meta['dataset_index']) + "_ann.json"
            dumpAnnotations(gt_path, 
                image_id=meta['dataset_index'], 
                image_name=meta['file_name'], 
                anns=ground_truths)

            pifpaf_cocoeval = metric.Coco(coco.COCO(gt_path), 
                max_per_image=10, 
                category_ids=range(1, len(datamodule.VisPanelModule.categories)+1), 
                iou_type='bbox')

            pifpaf_cocoeval.accumulate(pred, meta)

            coco_stats.append(pifpaf_cocoeval.stats()['stats'])

            if paint:
                with open(os.path.join(open_image_dir, meta['file_name']), 'rb') as f:
                    pil_image = PIL.Image.open(f).convert('RGB')
                visualizer.Base.image(pil_image)

                with show.image_canvas(pil_image, os.path.join(out_image_dir, meta['file_name'])) as ax:
                    annotation_painter.annotations(ax, pred)

                gt = []
                for ann in ground_truths:
                    annc = annotation.AnnotationDet(datamodule.VisPanelModule.categories)
                    annc.set(ann['category_id'], 1.0, ann['bbox'])
                    gt.append(annc)

                with show.image_canvas(pil_image, os.path.join(out_image_dir_gt, meta['file_name'])) as ax:
                    annotation_painter.annotations(ax, gt)

    return coco_stats

def stats_aggregate(coco_stats):
    aggregates = [0.0] * len(metric.Coco.text_labels_bbox)
    for stats in coco_stats:
        for i in range(len(aggregates)):
            aggregates[i] += stats[i]
    for i in range(len(aggregates)):
        aggregates[i] /= len(coco_stats)

    return aggregates

def print_eval(data_loader, index, foreword, paint=False, open_image_dir=tmp_image_dir):
    # block_print()

    coco_stats = collect(data_loader, file_name_indicator=str(index), paint=paint, open_image_dir=open_image_dir)
    coco_stats = stats_aggregate(coco_stats)

    # enable_print()

    print()
    print("=== AGGREGATED RESULTS: " + foreword + " ===")

    labels = [
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:0.3f}",
        "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:0.3f}",
        "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:0.3f}",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:0.3f}",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:0.3f}",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:0.3f}",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:0.3f}",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:0.3f}",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:0.3f}",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:0.3f}",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:0.3f}",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:0.3f}"
    ]
    for label, value in zip(labels, coco_stats):
        print(label.format(value))

    print()

preprocessor = get_preprocess()
# print_eval(custom_data_loader(10, preprocessor), index=0, foreword="from data loader", paint=True)
print_eval(generic_data_loader(wild_images_dir, 1, preprocessor), 
    index=0, 
    foreword="from images in the wild", 
    paint=True, 
    open_image_dir=wild_images_dir)

# def collect(data_loader):
#     image_names = []
#     image_sizes = []
#     gts = []

#     pifpafCoco = metric.Coco(None, max_per_image=10, category_ids=range(1, len(datamodule.VisPanelModule.categories)+1), iou_type='bbox')

#     processor, model = get_processor()
#     preprocess = get_preprocess()

#     # visualizers
#     annotation_painter = show.AnnotationPainter()

#     for batch_i, (image_tensors_batch, gts_batch, meta_batch) in enumerate(data_loader):
#         pred_batch = processor.batch(model, image_tensors_batch, device=device)

#         # unbatch
#         for pred, ground_truths, meta, image in zip(pred_batch, gts_batch, meta_batch, image_tensors_batch):
#             pred = preprocess.annotations_inverse(pred, meta) #array of annotations

#             for i, ann in enumerate(ground_truths):
#                 ann['image_id'] = meta['dataset_index']
#                 ann['bbox'] = ann['bbox_original'].tolist()
#                 ann['keypoints'] = ann['keypoints'].tolist()
#                 ann['id'] = i
#                 del ann['bbox_original']

#                 gts.append(ann)

#             pifpafCoco.accumulate(pred, meta)

#             print(meta['image_id'])

#             image_names.append(meta['file_name'])
#             image_sizes.append(meta['size'])

#             # # image output
#             with open(os.path.join(tmp_image_dir, meta['file_name']), 'rb') as f:
#                 pil_image = PIL.Image.open(f).convert('RGB')
#             visualizer.Base.image(pil_image)

#             with show.image_canvas(pil_image, os.path.join(out_image_dir, meta['file_name'])) as ax:
#                 annotation_painter.annotations(ax, pred)

#             gt = []
#             for ann in ground_truths:
#                 annc = annotation.AnnotationDet(datamodule.VisPanelModule.categories)
#                 annc.set(ann['category_id'], 1.0, ann['bbox'])
#                 gt.append(annc)

#             with show.image_canvas(pil_image, os.path.join(out_image_dir_gt, meta['file_name'])) as ax:
#                 annotation_painter.annotations(ax, gt)

#     dumpAnnotations(gt_tmp_dir, image_names, image_sizes, gts)

#     pifpafCoco.coco = pycocotools.coco.COCO(gt_tmp_dir)
#     return pifpafCoco