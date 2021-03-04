import os
import sys #for maxsize

import glob
import random
import json

import math # rotation cos, sin
from operator import le, ge # to pass < and > as arg

import copy
import logging
import numpy as np
import torch.utils.data
from collections import defaultdict

from PIL import Image, ImageEnhance
from openpifpaf import transforms, utils

import cv2 as cv

from .annotation_fields import AnnotationFields

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))

class VisPanel(torch.utils.data.Dataset):
    # we load categs directly from the annotation
    def load_cp_categories(self, annotations_dict):
        for category in annotations_dict[AnnotationFields.CATEGORIES]:
            self.category_names.append(category[AnnotationFields.CATEGORY_NAME])
            self.category_ids.append(category[AnnotationFields.CATEGORY_ID])

    def load_cp_images(self, annotations_dict):
        self.cp_image_names = [None] * len(annotations_dict[AnnotationFields.IMAGES])
        for image in annotations_dict[AnnotationFields.IMAGES]:
            self.cp_image_names[image[AnnotationFields.IMAGE_ID]-1] = image[AnnotationFields.IMAGE_NAME]

    def load_cp_annotations(self, annotations_dict):
        for it in self.cp_image_names:
            self.cp_annotations.append([])
        for annotation in annotations_dict[AnnotationFields.ANNOTATIONS]:
            img_index = annotation[AnnotationFields.ANNOTATION_IMAGE_ID] - 1
            # print(img_index)

            for field in AnnotationFields.ANNOTATION_FIELD_TO_DELETE:
                del annotation[field]

            self.cp_annotations[img_index].append(annotation)

    """
    tmpDir specified if we want to save the image we have obtained.
    Is is most useful to debug and to run a prediction: we only have to get image
    from disk instead of having to translate a pytorch image tensor to a PIL image.
    """
    def __init__(self, cp_image_dir, coco_image_dir, annotation_file, *,
                 n_images=None, preprocess=None,
                 category_ids=None, prediction=False, tmpDir=None, randomSeed=None):
        self.category_names = []
        self.category_ids = []

        self.cp_image_names = []
        self.cp_image_ids = []

        self.cp_annotations = []

        self.cp_image_dir = cp_image_dir
        self.coco_images = glob.glob(os.path.join(coco_image_dir, "*.jpg"))

        with open(annotation_file, 'r') as json_raw:
            annotations_dict = json.load(json_raw)

            self.load_cp_categories(annotations_dict)
            self.load_cp_images(annotations_dict)
            self.load_cp_annotations(annotations_dict)

        # PifPaf
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

        self.n_images = n_images

        self.prediction = prediction
        self.tmpDir = tmpDir

        # used for having a reliable stack of transforms when testing
        if randomSeed: 
            random.seed(randomSeed)

    def scale_down_if_need(self, image, max_width, max_height):
        width, height = image.size
        ratio = min(max_width/width, max_height/height)
        if ratio >= 1: 
            return image, 1.0, 1.0
        else:
            new_width = int(ratio*width)
            new_height = int(ratio*height)
            return image.resize((new_width, new_height), resample=Image.ANTIALIAS), new_width/width, new_height/height

    def rescale_annotation(self, cp_annotation, x_ratio, y_ratio):
        cp_annotation["bbox"][0] *= x_ratio
        cp_annotation["bbox"][1] *= y_ratio
        cp_annotation["bbox"][2] *= x_ratio
        cp_annotation["bbox"][3] *= y_ratio

        cp_annotation["area"] = cp_annotation["bbox"][2] * cp_annotation["bbox"][3]

    def shift_annotation(self, cp_annotation, x_offset, y_offset):
        cp_annotation["bbox"][0] += x_offset
        cp_annotation["bbox"][1] += y_offset

    def get_rand_offset(self, base_w, base_h, overlay_w, overlay_h):
        x_offset = random.uniform(0, base_w - overlay_w)
        y_offset = random.uniform(0, base_h - overlay_h)

        # returns the floored offsets as int
        return int(x_offset), int(y_offset)

    """
    Gets offset such that after having offset overlay, it won't go of from the background. 
    """
    def get_rand_offset_satisfying(self, background, overlay):
        overlay_width, overlay_height = overlay.size
        background_width, background_height = background.size

        return self.get_rand_offset(background_width, background_height, overlay_width, overlay_height)

    """ 
    modifies background AND returns it. 
    Why? More elegant way of getting a new name variable which holds the combination of images
    """
    def overlay_image(self, background, overlay, x_offset, y_offset):
        # return Image.alpha_composite(background, overlay)
        background.alpha_composite(overlay, (x_offset, y_offset))
        return background

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

    """
    Some op are faster with openCV, hence translating the image might be worth it.
    """
    def to_CV_format(self, pil_image):
        cv_image = np.array(pil_image)
        # convert from 3 to 4 channels.
        return cv.cvtColor(cv_image, cv.COLOR_RGB2BGRA) 

    """
    Some op are faster with openCV, hence translating the image might be worth it.
    Of course we need to translate them back with toPILFormat.
    """
    def to_PIL_format(self, cv_image):
        res = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        return Image.fromarray(res)

    def noise_transformed(self, image):
        def add_noise_at(x, y, std=5):
            def add_noise(v):
                val = int(v+random.normalvariate(0,std))
                return min(max(0, val), 255)
            
            r, g, b, a = image.getpixel((x,y))
            r = add_noise(r)
            g = add_noise(g)
            b = add_noise(b)
            image.putpixel((x,y), (r, g, b))

        if random.choice([True, False]):
            std = random.uniform(10, 100)
            w, h = image.size
            for x in range(w):
                for y in range(h):
                    add_noise_at(x, y, std=std)

        return image
        

    def color_transformed(self, image):
        def shift_hue(image, delta_hue):
            width, height = image.size
            image = image.convert('HSV')

            for x in range(width):
                for y in range(height):
                    hue, saturation, value = image.getpixel((x,y))      
                    image.putpixel((x,y), ((hue + delta_hue)%255, saturation, value))

            return image.convert('RGB')

        delta_hue = random.randint(0, 255)
        image = shift_hue(image, delta_hue)

        # Brightness
        factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    """ 
    Allows us to choose 4 pairs (image, preimage) in prespective transform 
    instead of obscure transformation coefficient
    """
    def get_coeffs(self, source_coords, target_coords):
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        A = np.matrix(matrix, dtype=np.float)
        B = np.array(source_coords).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    """
    The prespective transform is a kind of transform where we can choose 4 pairs of (preimage, image) 
    Where image is the image of preimage by the transform.

    More specifically, we use getCoeffs to choose 4 pairs, instead of chosing obscure coefficients.
    (that we don't know the meaning of)
    """
    def perspective_transformed(self, image):
        width, height = image.size

        k1 = random.uniform(0, 0.3)
        k2 = random.uniform(0.7, 1)
        choice = (random.choice(["top_left", "bottom_left"]), random.choice(["top_right", "bottom_right"]))

        domain = [(0, 0), (width, 0), (width, height), (0, height)]
        if choice == ("top_left", "top_right"):
            codomain = [(k1*width, 0), (k2*width, 0), (width, height), (0, height)]
        elif choice == ("bottom_left", "bottom_right"):
            codomain = [(0, 0), (width, 0), (k2*width, height), (k1*width, height)]
        elif choice == ("bottom_left", "top_right"):
            codomain = [(0, 0), (k2*width, 0), (width, height), (k1*width, height)]
        elif choice == ("top_left", "bottom_right"):
            codomain = [(k1*width, 0), (width, 0), (k2*width, height), (0, height)]
        coeffs = self.get_coeffs(domain, codomain)
        inv_coeffs = self.get_coeffs(codomain, domain)

        # print(choice)
        # print(k1, k2)

        res = image.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(255,255,255,0))
        return res, inv_coeffs

    """
    The transform we apply on the image is the perspective transform, 
    which map every point (x,y) to a point (x', y') such that:
        x' = (ax + by + c) / (gx + hy + 1)
        y' = (dx + ey + f) / (gx + hy + 1)
    """
    def perspective_transform_function(self, point, coeffs):
        (x, y) = point
        (a, b, c, d, e, f, g, h) = coeffs

        xt = (a*x + b*y + c) / (g*x + h*y + 1)
        yt = (d*x + e*y + f) / (g*x + h*y + 1)
    
        return (xt, yt)

    """
    We can use the same formula as the perspective transform with the annotations.

    For now: we use the naive implementation of taking the biggest bounding box which contains
    the transform of the previous bounding box.

    Given our implementation, y and height never change though.
    """
    def perspective_transform_annotation(self, annotation, coeffs):
        bbox = annotation["bbox"]
        x = bbox[0]; y = bbox[1]; w = bbox[2]; h = bbox[3]

        topLeft = (x, y)
        topRight = (x+w, y)
        bottomLeft = (x, y+h)
        bottomRight = (x+w, y+h)
        
        newTopLeft = self.perspective_transform_function(topLeft, coeffs)
        newTopRight = self.perspective_transform_function(topRight, coeffs)
        newBottomLeft = self.perspective_transform_function(bottomLeft, coeffs)
        newBottomRight = self.perspective_transform_function(bottomRight, coeffs)

        xMin = min(newTopLeft[0], newBottomLeft[0])
        xMax = max(newTopRight[0], newBottomRight[0])

        """
        the min/max arguments should be the same given how the transform is implemented, 
        but the following would be correct under any perspective transform,
        not just the subset of transforms we are applying
        """
        yMin = min(newTopLeft[1], newTopRight[1])
        yMax = max(newBottomLeft[1], newBottomRight[1])

        annotation["bbox"][0] = xMin
        annotation["bbox"][1] = yMin
        annotation["bbox"][2] = abs(xMax-xMin)
        annotation["bbox"][3] = abs(yMax-yMin)

    """
    Apply rotation to image and return the transformed image along with the chosen angle
    """
    def rotation_transformed(self, image):
        width, height = image.size

        angle = random.uniform(0, 360)

        # print(f"angle: {angle}")

        return image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255,255,255,0)), angle, image.size

    """
    For any point (x,y), get the projection of the point under the rotation transform
    """
    def rotation_function(self, point, angle, size):
        w, h = size
        rotn_center = (w / 2.0, h / 2.0)

        angle = math.radians(angle)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(-rotn_center[0], -rotn_center[1], matrix)

        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]

        # calculate output size to offset the new point after transform.
        xx = []
        yy = []
        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            x, y = transform(x, y, matrix)
            xx.append(x)
            yy.append(y)
        nw = math.ceil(max(xx)) - math.floor(min(xx))
        nh = math.ceil(max(yy)) - math.floor(min(yy))

        size_delta = (nw - w) / 2.0, (nh - h) / 2.0

        mapsto = transform(point[0], point[1], matrix)
        return mapsto[0] + size_delta[0], mapsto[1] + size_delta[1]

    '''
    Apply the rotation transform with param angle to an annotation.
    In particular, we compute the largest bounding box which contains the mapped rectangle.
    '''
    def rotate_transform_annotation(self, annotation, angle, size):
        def compPoints(points, index, compFun, coeff):
            extremum = coeff*sys.maxsize
            for point in points:
                if(compFun(point[index], extremum)):
                    extremum = point[index]
            return extremum

        def minPoints(points, index):
            return compPoints(points, index, le, 1)

        def maxPoints(points, index):
            return compPoints(points, index, ge, -1)

        bbox = annotation["bbox"]
        x = bbox[0]; y = bbox[1]; w = bbox[2]; h = bbox[3]

        topLeft = (x, y)
        topRight = (x+w, y)
        bottomLeft = (x, y+h)
        bottomRight = (x+w, y+h)
        
        newTopLeft = self.rotation_function(topLeft, angle, size)
        newTopRight = self.rotation_function(topRight, angle, size)
        newBottomLeft = self.rotation_function(bottomLeft, angle, size)
        newBottomRight = self.rotation_function(bottomRight, angle, size)

        points = [newTopLeft, newTopRight, newBottomLeft, newBottomRight]

        xMin = minPoints(points, 0)
        yMin = minPoints(points, 1)

        xMax = maxPoints(points, 0)
        yMax = maxPoints(points, 1)

        # print(f"x: {xMin}, y: {yMin}, w: {abs(xMax-xMin)}, h: {abs(yMax-yMin)}")

        annotation["bbox"][0] = xMin
        annotation["bbox"][1] = yMin
        annotation["bbox"][2] = abs(xMax-xMin)
        annotation["bbox"][3] = abs(yMax-yMin)
    """
    Args:
        index (int): Index
    Returns:
        tuple: Tuple (image, annotations)
    """
    def __getitem__(self, index):
        LOG.info("get item")
        # For ex, the first 3 images will have the same background
        coco_index = index // len(self.cp_image_names)
        cp_index = index % len(self.cp_image_names)

        cp_image_path = os.path.join(self.cp_image_dir, self.cp_image_names[cp_index])
        coco_image_path = self.coco_images[coco_index]
        with Image.open(cp_image_path) as f:
            # RGBA format used to overlay without black background instead of clear background
            # (see alpha_composite of PIL)
            cp_image = f.convert('RGBA') 

        with Image.open(coco_image_path) as f:
            # RGBA format used to overlay without black background instead of clear background
            # In particular, the background has to be RGBA for the alpha_composite to work
            # We re translate later to RGB. (because the model works with 3 channels.)
            coco_image = f.convert('RGBA')

        meta_init = {
            'dataset_index': index,
            'image_id': index,
            'file_dir': cp_image_path,
            'file_name': str(index) + os.path.basename(cp_image_path)
        }

        """
        We are doing 3 things with the image:

        1. Apply perspective transform to simulate looking at the panel from a weird angle.
        2. Scale down if needed (happens if it's biggers than the background, which is the image from the coco dataset)
        3. Overlay the image on top of an image from the coco dataset

        Of course, we keep the coefficients, ratio, offsets from each respective transform to apply them to each annotation later.
        """
        cp_image = self.noise_transformed(cp_image)

        cp_image, coefficients = self.perspective_transformed(cp_image)
        cp_image, angle, rot_size = self.rotation_transformed(cp_image)

        coco_width, coco_height = coco_image.size
        cp_image, x_ratio, y_ratio = self.scale_down_if_need(cp_image, int(coco_width*0.7), int(coco_height*0.7))

        x_offset, y_offset = self.get_rand_offset_satisfying(coco_image, cp_image)
        image = self.overlay_image(coco_image, cp_image, x_offset, y_offset)

        # We re translate later to RGB. (because the model works with 3 channels.)
        image = image.convert('RGB')
        image = self.color_transformed(image)

        # easier debugging and prediction by saving the image to disk.
        if self.tmpDir is not None:
            image.save(os.path.join(self.tmpDir, meta_init['file_name']))

        meta_init['size'] = image.size

        anns = []
        # even if we are not doing prediction, we want ground truth for metrics.
        for annotation in self.cp_annotations[cp_index]:
            annotation = copy.deepcopy(annotation)
            self.perspective_transform_annotation(annotation, coefficients)
            self.rotate_transform_annotation(annotation, angle, rot_size)
            self.rescale_annotation(annotation, x_ratio, y_ratio)
            self.shift_annotation(annotation, x_offset, y_offset)

            # adding missing field
            annotation["keypoints"] = []
            annotation["num_keypoints"] = 0
            annotation["segmentation"] = []

            anns.append(annotation)

        # self.print_debug_infos(anns, image, initial_cp_image, initial_coco_image, cp_image, x_offset, y_offset, x_ratio, y_ratio)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)

        LOG.info("image shape:")
        LOG.info(image.shape)

        return image, anns, meta

    def __len__(self):
        if self.n_images is not None:
            return self.n_images
        else: 
            return len(self.cp_image_names) * len(self.coco_images)
