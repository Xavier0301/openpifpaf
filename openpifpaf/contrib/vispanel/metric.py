import logging
import os
import numpy as np
import zipfile
import json
from tqdm import tqdm
from PIL import Image

from openpifpaf.metric.base import Base
from visdroneval.utils import calcAccuracy, dropObjectsInIgr

LOG = logging.getLogger(__name__)

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class VisDrone(Base):
    text_labels = ['AP_all', 'AP_50', 'AP_75', 'AR_1', 'AR_10',
                            'AR_100', 'AR_500']
    def readFile(self, filepath):
        result = []
        with open(filepath) as f:
            for line in f:
                result.append([float(elem) for elem in line.split(',')])

        return np.asarray(result)

    def gt_ignoreCleaning(self, gt_og, pred_og):
        #%% process the annotations and groundtruth
        allgt = []
        alldet = []

        for nameImg in tqdm(gt_og.keys()):

            oldgt = gt_og[nameImg]
            olddet = pred_og[nameImg]
            #% remove the objects in ignored regions or labeled as others

            with open(os.path.join(self.imgPath, nameImg[:-4]+'.jpg'), 'rb') as f:
                image = Image.open(f).convert('RGB')
                imgWidth, imgHeight = image.size
            newgt, det = dropObjectsInIgr(oldgt, olddet, imgHeight, imgWidth);

            gt = newgt;
            gt[newgt[:,4] == 0, 4] = 1;
            gt[newgt[:,4] == 1, 4] = 0;
            allgt.append(gt);
            alldet.append(sorted(det, key=lambda x: x[4], reverse=True));

        return np.asarray(allgt), np.asarray(alldet)
    def __init__(self, gt_dir, img_dir):

        self.imgPath = img_dir
        self.predictions = {}
        self.image_ids = []
        self.gt = {}
        for fileName in os.listdir(gt_dir):
            if not '.txt' in fileName:
                continue
            self.gt[fileName] = self.readFile(os.path.join(gt_dir, fileName))


    def accumulate(self, predictions, image_meta, ground_truth=None):
        image_id = int(image_meta['image_id'])
        width, height = image_meta['width_height']
        self.image_ids.append(image_id)

        image_annotations = []
        for pred in predictions:
            pred_data = pred.json_data()
            categ = pred_data['category_id']
            x,y,w,h = pred_data['bbox']
            x1, x2 = np.clip([x, x+w], a_min=0, a_max=width)
            y1, y2 = np.clip([y, y+h], a_min=0, a_max=height)
            s = pred_data['score']
            image_annotations.append([x1, y1, x2-x1, y2-y1, s, categ, -1, -1])

        self.predictions[image_meta['file_name'][:-4]+'.txt'] = np.asarray(image_annotations)


    def write_predictions(self, filename, additional_data=None):
        mkdir_if_missing(filename)
        for imageName in self.predictions.keys():
            with open(os.path.join(filename,imageName), "w") as file:
                file.write("\n".join([",".join(item) for item in self.predictions[imageName].astype(str)]))

        LOG.info('wrote predictions to %s', filename)
        with zipfile.ZipFile(os.path.join(filename,'predictions')+ '.zip', 'w') as myzip:
            for imageName in self.predictions.keys():
                myzip.write(os.path.join(filename,imageName))
        LOG.info('wrote %s.zip', os.path.join(filename,'predictions')+ '.zip')

        if additional_data:
            with open(os.path.join(filename, 'predictions.pred_meta.json'), 'w') as f:
                json.dump(additional_data, f)
            LOG.info('wrote %s.pred_meta.json', filename)


    def stats(self):
        allgt, alldet = self.gt_ignoreCleaning(self.gt, self.predictions)
        AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500, AP = calcAccuracy(len(alldet), allgt, alldet)

        data = {
            'stats': [AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500],
            'text_labels': self.text_labels,
        }

        return data
