import os

import glob
import random
import json

import copy
import logging
import numpy as np
import torch.utils.data
from collections import defaultdict

from PIL import Image
from openpifpaf import transforms, utils

from .annotation_fields import AnnotationFields

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))

class VisPanel(torch.utils.data.Dataset):
    def load_cp_categories(self, annotations_dict):
        for category in annotations_dict[AnnotationFields.CATEGORIES]:
            self.category_names.append(category[AnnotationFields.CATEGORY_NAME])
            self.category_ids.append(category[AnnotationFields.CATEGORY_ID])

    def load_cp_images(self, annotations_dict):
        self.cp_image_names = [None] * len(annotations_dict[AnnotationFields.IMAGES])
        for image in annotations_dict[AnnotationFields.IMAGES]:
            self.cp_image_names[image[AnnotationFields.IMAGE_ID]-1] = image[AnnotationFields.IMAGE_NAME]
        print("==========================================================================================")
        print("==========================================================================================")
        print("==========================================================================================")
        for i, image_name in enumerate(self.cp_image_names):
            print(f"image {i}: {image_name}")
            print("___")
        print("==========================================================================================")
        print("==========================================================================================")
        print("==========================================================================================")

    def load_cp_annotations(self, annotations_dict):
        for it in self.cp_image_names:
            self.cp_annotations.append([])
        for annotation in annotations_dict[AnnotationFields.ANNOTATIONS]:
            img_index = annotation[AnnotationFields.ANNOTATION_IMAGE_ID] - 1
            print(img_index)

            for field in AnnotationFields.ANNOTATION_FIELD_TO_DELETE:
                del annotation[field]

            self.cp_annotations[img_index].append(annotation)

        print("==========================================================================================")
        print("==========================================================================================")
        print("==========================================================================================")
        for i, annotation in enumerate(self.cp_annotations):
            print(f"image {i}: {len(annotation)} annotations")
            print("___")
        print("==========================================================================================")
        print("==========================================================================================")
        print("==========================================================================================")

    def __init__(self, cp_image_dir, coco_image_dir, annotation_file, *,
                 n_images=None, preprocess=None,
                 category_ids=None):
        self.category_names = []
        self.category_ids = []

        self.cp_image_names = []
        self.cp_image_ids = []

        self.cp_annotations = []

        self.cp_image_dir = cp_image_dir
        # self.cp_images = glob.glob(os.path.join(cp_image_dir, "*.jpg"))
        self.coco_images = glob.glob(os.path.join(coco_image_dir, "*.jpg"))

        with open(annotation_file, 'r') as json_raw:
            annotations_dict = json.load(json_raw)

            self.load_cp_categories(annotations_dict)
            self.load_cp_images(annotations_dict)
            self.load_cp_annotations(annotations_dict)

        # PifPaf
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def scale_down_if_need(self, image, max_width, max_height):
        width, height = image.size
        ratio = min(max_width/width, max_height/height)
        if ratio >= 1: 
            return image, 1.0, 1.0
        else:
            new_width = int(ratio*width)
            new_height = int(ratio*height)
            return image.resize((new_width, new_height)), new_width/width, new_height/height

    def rescale_annotation(self, cp_annotation, x_ratio, y_ratio):
        cp_annotation["bbox"][0] *= x_ratio
        cp_annotation["bbox"][1] *= y_ratio
        cp_annotation["bbox"][2] *= x_ratio
        cp_annotation["bbox"][3] *= y_ratio

        cp_annotation["area"] = cp_annotation["bbox"][2] * cp_annotation["bbox"][3]

    def get_rand_offset(self, base_w, base_h, overlay_w, overlay_h):
        x_offset = random.uniform(0, base_w - overlay_w)
        y_offset = random.uniform(0, base_h - overlay_h)

        # returns the floored offsets as int
        return int(x_offset), int(y_offset)

    def get_rand_offset_satisfying(self, background, overlay):
        overlay_width, overlay_height = overlay.size
        background_width, background_height = background.size

        return self.get_rand_offset(background_width, background_height, overlay_width, overlay_height)

    # modifies background AND returns it. Why? No need to copy + more elegant way of getting a new name variable which holds background
    def overlay_image(self, background, overlay, x_offset, y_offset):
        background.paste(overlay, (x_offset, y_offset))
        return background

    def adjust_annotation(self, cp_annotation, x_offset, y_offset):
        cp_annotation["bbox"][0] += x_offset
        cp_annotation["bbox"][1] += y_offset

    def print_debug_infos(self, annotations, final_im, original_cp_im, coco_im, cp_im, x_offset, y_offset, x_ratio, y_ratio):
        print("==========================================================================================")
        print("==========================================================================================")
        print("==========================================================================================")
        print(f"""Image size: ${final_im.size}, background: ${coco_im.size}, foreground: ${cp_im.size} \n 
                foreground at scale: ${original_cp_im.size} \n 
                x_offset: ${x_offset}, y_offset: ${y_offset} \n 
                x_ratio: ${x_ratio}, y_ratio: ${y_ratio}""")
        print("==========================================================================================")
        print("==========================================================================================")
        print("==========================================================================================")
        for i, annotation in enumerate(annotations):
            print(f"$annotation {i} bbox: ${annotation['bbox']}")
            print("________")
        print("==========================================================================================")
        print("==========================================================================================")
        print("==========================================================================================")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco_index = index // len(self.cp_image_names)
        cp_index = index % len(self.cp_image_names)

        cp_image_path = os.path.join(self.cp_image_dir, self.cp_image_names[cp_index])
        coco_image_path = self.coco_images[coco_index]
        with Image.open(cp_image_path) as f:
            cp_image = f.convert('RGB')

        with Image.open(coco_image_path) as f:
            coco_image = f.convert('RGB')

        initial_size = cp_image.size
        meta_init = {
            'dataset_index': index,
            'image_id': (cp_image_path, coco_image_path),
            'file_dir': cp_image_path,
            'file_name': os.path.basename(cp_image_path),
        }
        # meta_init = {
        #     'dataset_index': index,
        #     'image_id': index,
        #     'cp_file_dir': cp_image_path,
        #     'coco_file_dir': coco_image_path
        #     'cp_file_name': os.path.basename(cp_image_path),
        #     'coco_file_name': os.path.basename(coco_image_path),
        # }
        initial_cp_image = cp_image
        initial_coco_image = coco_image

        coco_width, coco_height = coco_image.size
        cp_image, x_ratio, y_ratio = self.scale_down_if_need(cp_image, int(coco_width*0.7), int(coco_height*0.7))

        x_offset, y_offset = self.get_rand_offset_satisfying(coco_image, cp_image)
        image = self.overlay_image(coco_image, cp_image, x_offset, y_offset)

        anns = []
        for annotation in self.cp_annotations[cp_index]:
            annotation = copy.deepcopy(annotation)
            self.rescale_annotation(annotation, x_ratio, y_ratio)
            self.adjust_annotation(annotation, x_offset, y_offset)

            # adding missing field
            annotation["keypoints"] = []
            annotation["num_keypoints"] = 0
            annotation["segmentation"] = []

            anns.append(annotation)

        self.print_debug_infos(anns, image, initial_cp_image, initial_coco_image, cp_image, x_offset, y_offset, x_ratio, y_ratio)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)

        # transform image

        # mask valid
        # valid_area = meta['valid_area']
        # utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        # LOG.debug(meta)

        # # log stats
        # for ann in anns:
        #     if getattr(ann, 'iscrowd', False):
        #         continue
        #     if not np.any(ann['keypoints'][:, 2] > 0.0):
        #         continue
        #     STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})


        return image, anns, meta

    def __len__(self):
        return len(self.cp_image_names) * len(self.coco_images)
