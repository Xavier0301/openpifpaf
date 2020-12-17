"""Video demo application.

Use --scale=0.2 to reduce the input image size to 20%.
Use --json-output for headless processing.

Example commands:
    python3 -m pifpaf.video --source=0  # default webcam
    python3 -m pifpaf.video --source=1  # another webcam

    # streaming source
    python3 -m pifpaf.video --source=http://127.0.0.1:8080/video

    # file system source (any valid OpenCV source)
    python3 -m pifpaf.video --source=docs/coco/000000081988.jpg

Trouble shooting:
* MacOSX: try to prefix the command with "MPLBACKEND=MACOSX".
"""


import argparse
import json
import logging
import os
import time

import numpy as np
import torch

import pyrealsense2 as rs

import cv2  # pylint: disable=import-error
from . import decoder, logger, network, plugins, show, transforms, visualizer, __version__

try:
    import mss
except ImportError:
    mss = None

LOG = logging.getLogger(__name__)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

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

device = torch.device('cpu')

from openpifpaf import datasets, decoder, logger, network, plugins, metric, show, transforms, visualizer, annotation,__version__

from torchvision import transforms as torchTransforms

import pycocotools 

# from vispanel import VisPanel
# from datamodule import VisPanelModule
import vispanel 
import datamodule

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

def get_processor():
    model_name = 'vispanel-lr-small.pkl'
    model_path = '/Users/xavier/Desktop/EPFL_Cours/Ici/ERC/object_detection/models/' + model_name
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

def rs_pipeline_init():
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    pipeline.start()

    return pipeline


# pylint: disable=too-many-branches,too-many-statements
def main(scale=1.0):
    processor, model = get_processor()
    preprocess = get_preprocess()

    # create keypoint painter
    annotation_painter = show.AnnotationPainter()

    animation = show.AnimationFrame()
    last_loop = time.time()
    for frame_i, (ax, ax_second) in enumerate(animation.iter()):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame: 
            continue 

        # you'll maybe have to recolor idk
        image = np.asanyarray(color_frame.get_data())

        start = time.time()
        if scale != 1.0:
            image = cv2.resize(image, None, fx=args.scale, fy=args.scale)

        if ax is None:
            ax, ax_second = animation.frame_init(image)
        visualizer.Base.image(image)
        visualizer.Base.common_ax = ax_second

        image_pil = PIL.Image.fromarray(image)
        meta = {
            'hflip': False,
            'offset': np.array([0.0, 0.0]),
            'scale': np.array([1.0, 1.0]),
            'valid_area': np.array([0.0, 0.0, image_pil.size[0], image_pil.size[1]]),
        }
        processed_image, _, meta = preprocess(image_pil, [], meta)
        visualizer.Base.processed_image(processed_image)
        LOG.debug('preprocessing time %.3fs', time.time() - start)

        preds = processor.batch(model, torch.unsqueeze(processed_image, 0), device=device)[0]

        start_post = time.perf_counter()
        preds = preprocess.annotations_inverse(preds, meta)

        if args.json_output:
            with open(args.json_output, 'a+') as f:
                json.dump({
                    'frame': frame_i,
                    'predictions': [ann.json_data() for ann in preds]
                }, f, separators=(',', ':'))
                f.write('\n')
        if not args.json_output or args.video_output:
            ax.imshow(image)
            annotation_painter.annotations(ax, preds)

        LOG.debug('time post = %.3fs', time.perf_counter() - start_post)
        LOG.info('frame %d, loop time = %.3fs, FPS = %.3f',
                 frame_i,
                 time.time() - last_loop,
                 1.0 / (time.time() - last_loop))
        last_loop = time.time()

        if args.max_frames and frame_i >= args.max_frames:
            break


if __name__ == '__main__':
    main()
