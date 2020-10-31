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

    category_names = []
    category_ids = []

    cp_image_names = []
    cp_image_ids = []

    cp_annotations = []

    def load_cp_categories(self, annotations_dict):
        for category in annotations_dict[AnnotationFields.CATEGORIES]:
            self.category_names.append(category[AnnotationFields.CATEGORY_NAME])
            self.category_ids.append(category[AnnotationFields.CATEGORY_ID])

    def load_cp_images(self, annotations_dict):
        for image in annotations_dict[AnnotationFields.IMAGES]:
            self.cp_image_names.append(image[AnnotationFields.IMAGE_NAME])
            self.cp_image_ids.append(image[AnnotationFields.IMAGE_ID])

    def load_cp_annotations(self, annotations_dict):
        for image_id in self.cp_image_ids:
            self.cp_annotations.append([])
        for annotation in annotations_dict[AnnotationFields.ANNOTATIONS]:
            img_index = annotation[AnnotationFields.ANNOTATION_IMAGE_ID] - 1

            for field in AnnotationFields.ANNOTATION_FIELD_TO_DELETE:
                del annotation[field]

            self.cp_annotations[img_index].append(annotation)

    def __init__(self, cp_image_dir, coco_image_dir, annotation_file, *,
                 n_images=None, preprocess=None,
                 category_ids=None):
        self.cp_images = glob.glob(os.path.join(cp_image_dir, "*.jpg"))
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
        if(width > max_width):
            ratio = max_width / width
            new_height = int(ratio * height)
            if(new_height <= max_height):
                return image.resize((max_width, new_height)), ratio

        if(height > max_height):
            ratio = max_height / height
            new_width = int(ratio * width)
            return image.resize((new_width, max_height)), ratio

        return image, 1.0

    def rescale_annotation(self, cp_annotation, ratio):
        w = cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_WIDTH]
        h = cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_HEIGHT]

        cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_WIDTH] = ratio*w
        cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_HEIGHT] = ratio*h

        x = cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_X]
        y = cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_Y]

        cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_X] = x*ratio
        cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_Y] = y*ratio

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
        x = cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_X]
        y = cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_Y]

        cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_X] = x + x_offset
        cp_annotation[AnnotationFields.ANNOTATION_BBOX][AnnotationFields.ANNOTATION_BBOX_Y] = y + y_offset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco_index = index // len(self.cp_images)
        cp_index = index % len(self.cp_images)

        cp_image_path = self.cp_images[cp_index]
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

        coco_width, coco_height = coco_image.size
        cp_image, ratio = self.scale_down_if_need(cp_image, int(coco_width*0.9), int(coco_height*0.9))

        x_offset, y_offset = self.get_rand_offset_satisfying(coco_image, cp_image)
        image = self.overlay_image(coco_image, cp_image, x_offset, y_offset)

        anns = []
        for annotation in self.cp_annotations[cp_index]:
            self.rescale_annotation(annotation, ratio)
            self.adjust_annotation(annotation, x_offset, y_offset)

            # adding missing field
            annotation["keypoints"] = []
            annotation["num_keypoints"] = 0
            annotation["segmentation"] = []

            anns.append(annotation)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)

        # transform image

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        LOG.debug(meta)

        # # log stats
        # for ann in anns:
        #     if getattr(ann, 'iscrowd', False):
        #         continue
        #     if not np.any(ann['keypoints'][:, 2] > 0.0):
        #         continue
        #     STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})


        return image, anns, meta

    def __len__(self):
        return len(self.cp_images) * len(self.coco_images)
