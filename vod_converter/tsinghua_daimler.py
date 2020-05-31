"""
Ingestor for Tsinghua Daimler formats.

http://www.gavrila.net/data/Daimler/iv16-li-flohr-gavrila-etal/README_TDCB.pdf

Per devkit docs:

Within this dataset each frame is represented by one json-file with a predefined format. The name of
the json-file follows the scheme: tsinghuaDaimlerDataset_{seq:0>6}_{frame:0>9}_{type}.json
Possible values of type:
- labelData: Annotated bounding box data
- detections: Detections represented by bounding boxes produced by YOUR alogirthm
Attention: During the evaluation process the correspondence is performed based on the filename of
the detection and ground truth files.


Ground truth annotation and detection fields:

#Line number                           Variable name                Meaning
--------------------------------------------------------------------------------------------------
  mincol, minrow, maxcol, maxrow       [unsigned integer]           Defining the position and size
                                                                    of the bouding box by its upper
                                                                    left and bottom right point    
                                             
  identity                             [string]                     Specifies the object class.
                                                                    Possible values are
                                                                    ["unlabeled", "pedestrian",
                                                                    "cyclist", "motorcyclist",
                                                                    "tricyclist", "wheelchairuser",
                                                                    "mopedrider"]

  tags                                 [list of strings]            List (possible empty) of
                                                                    specified tags. Used during the
                                                                    evaluation (occlusion level
                                                                    only). Possible tags are
                                                                    ["out_of_image",
                                                                    "unsure_box", "occluded>10",
                                                                    "occluded>40", "occluded>80"]
Ground truth only fields

#Line number                           Variable name                Meaning
--------------------------------------------------------------------------------------------------
  uniqueid                             [integer]                    Unique number for each
                                                                    ground truth annotation,
                                                                    specifies one annotated object
                                                                    at one specific time  
                                             
  trackid                              [string]                     Unique string for each real
                                                                    object, consistent over the
                                                                    whole dataset (Layout:
                                                                    "{identity}_{0>6}")                                                                                            
Detection only fields

#Line number                           Variable name                Meaning
--------------------------------------------------------------------------------------------------
  score                                [float]                       Detection score (confidence) of
                                                                    the given detection, higher is
                                                                    better
"""

import json
import os
from PIL import Image
import shutil

from converter import Ingestor, Egestor


class TsinghuaDaimlerIngestor(Ingestor):
    def __init__(self):
        self.sub_dirs = ['train', 'valid', 'test']
    
    def validate(self, path):
        expected_dirs = [
            'labelData/train/tsinghuaDaimlerDataset',
            'leftImg8bit/train/tsinghuaDaimlerDataset',
            'labelData/valid/tsinghuaDaimlerDataset',
            'leftImg8bit/valid/tsinghuaDaimlerDataset',
            'labelData/test/tsinghuaDaimlerDataset',
            'leftImg8bit/test/tsinghuaDaimlerDataset'
        ]
        for subdir in expected_dirs:
            if not os.path.isdir(f"{path}/{subdir}"):
                return False, f"Expected subdirectory {subdir} within {path}"
        return True, None

    # Filename example: tsinghuaDaimlerDataset_2015-03-24_041424_000029321_leftImg8bit.png
    def get_image_path(self, root, sub_dir, image_id, image_ext):
        return f"{root}/leftImg8bit/{sub_dir}/tsinghuaDaimlerDataset/{image_id}_leftImg8bit.{image_ext}"

    # Filename example: tsinghuaDaimlerDataset_2015-03-20_025811_000034282_labelData.json
    def get_label_path(self, root, sub_dir, image_id):
        return f"{root}/labelData/{sub_dir}/tsinghuaDaimlerDataset/{image_id}_labelData.json"
    
    def ingest(self, path):
        sub_dir_with_image_ids = self._get_image_ids(path)
        image_ext = 'png'
        results = []
        for sub_dir, image_ids in sub_dir_with_image_ids:
            for image_id in image_ids:
                results.append(self._get_image_detection(path, sub_dir, image_id, image_ext))
        return results

    # Run cs231n_project/vod-converter/preprocess_cyclist_data.ipynb before geetting the image ids.
    # Return list of tuple (image_subdir, image_id)
    # e.g. [('train', 001.png'), ('test', '002.png')]
    def _get_image_ids(self, root):
        ret = []
        for sub_dir in self.sub_dirs:
            path = f"{root}/{sub_dir}.txt"
            with open(path) as f:
                image_ids = f.read().strip().split('\n')
                for i in range(len(image_ids)):
                    image_ids[i] = image_ids[i][:-len('_leftImg8bit')]
                ret.append((sub_dir, image_ids))
        return ret
        

    def _get_image_detection(self, root, sub_dir, image_id, image_ext='png'):
        image_path = self.get_image_path(root, sub_dir, image_id, image_ext)
        image_width, image_height = _image_dimensions(image_path)
        detections_fpath = self.get_label_path(root, sub_dir, image_id)
        detections = self._get_detections(detections_fpath, image_width, image_height)
        detections = [det for det in detections if det['left'] < det['right'] and det['top'] < det['bottom']]
        return {
            'image': {
                'id': image_id,
                'path': image_path,
                'segmented_path': None,
                'width': image_width,
                'height': image_height
            },
            'detections': detections
        }

    def _get_detections(self, detections_fpath, image_width, image_height):
        detections = []
        with open(detections_fpath) as f:
            f_json = json.load(f)
            for box in f_json['children']:
                label = box['identity']
                min_col = float(box['mincol'])
                max_col = float(box['maxcol'])
                min_row = float(box['minrow'])
                max_row = float(box['maxrow'])
                if min_col < 0:
                    # print(f'Fixed detection out range, min_col=: {min_col}')
                    min_col = 0
                if max_col >= image_width:
                    # print(f'Fixed detection out range, max_col=: {max_col}')
                    max_col = image_width-1
                if min_row < 0:
                    # print(f'Fixed detection out range, min_row=: {min_row}')
                    min_row = 0
                if max_row >= image_height:
                    # print(f'Fixed detection out range, max_row=: {max_row}')
                    max_row = image_height-1
                detections.append({
                    'label': label,
                    'left': min_col,
                    'right': max_col,
                    'top': min_row,
                    'bottom': max_row
                })
        return detections


def _image_dimensions(path):
    with Image.open(path) as image:
        return image.width, image.height