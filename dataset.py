import os
import json
import re
import shutil

import numpy as np
import cv2

import math
import random
import copy

from . annotation.shapes import Vialiboundingbox, Vialibpolygon
from . visualisation.visualizer import VisualizerPolygons, VisualizerBoundingBoxes
from . converter.format import ConverterBoundinBox, ConverterPolygons
from . augmentation.augmenter import AugmenterBoundingBox, AugmenterPolygon

class DatasetPolygon:
    """ Dataset Class

    author: Ashry Ramadhan
    """
    
    # Basic members
    __dataset = None
    __dataset_dict = {}
    __via = None
    length = 0
    __input_dir = None
    __output_dir = None

    # Object class
    class_list = []

    # __Polygon
    __Polygon = None
    __Polygon_data = {}

    # Visualizer
    __DatasetVisualizer = None

    # Converter
    __Converter = None

    # Augmenter
    __Augmenter = None

    def __init__(self, images_directory, output_directory, annotation_file_name):

        json_file = os.path.join(images_directory, annotation_file_name)
        with open(json_file) as f:
            imgs_ann = json.load(f)
        
        # clean img_anns from non-annotated images
        k_sus = []
        for k, v in imgs_ann.items():
            if len(v['regions']) == 0:
                k_sus.append(k)

        for k in k_sus:
            del imgs_ann[k]

        # creating dataset json
        dataset_list = []
        for idx, v in enumerate(imgs_ann.values()):

            filename = os.path.join(images_directory, v["filename"])

            if cv2.imread(filename) is not None:
                
                height, width = cv2.imread(filename).shape[:2]
                
                record = {}
                record["file_name"] = filename
                record["key"] = v["filename"] + str(v["size"])
                record["image_id"] = idx
                record["height"] = height
                record["width"] = width
            
                annos = v["regions"]
                objs = []
                if len(annos) > 0:
                    for anno in annos:
                        reg = anno["region_attributes"]
                        anno = anno["shape_attributes"]
                        px = anno["all_points_x"]
                        py = anno["all_points_y"]
                        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                        poly = [p for x in poly for p in x]
                        region_class = reg["type"]
                        obj = {
                            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                            "segmentation": [poly],
                            "all_points_x": px,
                            "all_points_y": py,
                            "class": region_class,
                            "category_id": 0,
                            "iscrowd": 0
                            }
                        objs.append(obj)
                    record["annotations"] = objs
                    dataset_list.append(record)
                    self.__dataset_dict[os.path.basename(filename)] = record

            self.__dataset = dataset_list
            self.__via = imgs_ann
            self.length = len(self.__dataset)
            self.__input_dir = images_directory
            self.__output_dir = output_directory

            if not os.path.exists(self.__output_dir):
                os.mkdir(self.__output_dir)

            # Polygon
            self.__Polygon = Vialibpolygon(self.__dataset)
            self.__Polygon_data, self.class_list = self.__Polygon.get_polygons()

            # Visualizer
            self.__DatasetVisualizer = VisualizerPolygons(self.__dataset)

            # Converter
            self.__Converter = ConverterPolygons(self.__dataset, self.class_list, self.__Polygon_data)

            # Augmenter
            self.__Augmenter = AugmenterPolygon(self.__dataset, self.__via, self.__Polygon_data)

    def printDataset(self):
        """Print dataset correspond to this object
        Args:
            None
        """
        for i in range(len(self.__dataset)):
            print(self.__dataset[i])

    def printClasses(self):
        """Print all classes correspond to this object
        Args:
            None
        """
        print(self.class_list)

    def plot_dataset(self, opacity=0.5, cols=2, image_start=0, num_of_sample=4, color_dict=None, without_bboxes=True, randomness=True):
        """Plot the dataset
        Args:
            opacity: opacity of alpha face
            cols: nb. fo columns in plot
            image_start: starting index if randomness is False
            num_of_sample: sample that need to be plotted
            color_dict: color dictionary for each object
            without_bboxes: plot without bounding boxes
            randomness: set True if you want to display your images randomly
        """
        self.__DatasetVisualizer.plot_dataset(polygons_data=self.__Polygon_data, opacity=opacity, cols=cols, image_start=image_start, num_of_sample=num_of_sample, color_dict=color_dict, without_bboxes=without_bboxes, randomness=randomness)

    def visualize_keypoints_of_image(self, image, polygons, color=(0, 255, 0), diameter=15):
        """Visualize the keypoints
        Args:
            image: image object (ndarray)
            polygons: image polygons
            color: keypoints color
            diameter: diameter of keypoints
        """
        self.__DatasetVisualizer.visualize_keypoints_of_image(image=image, polygons=polygons, color=color, diameter=diameter)

    ###########################  CONVERTER ############################################
    def convert_to_yolo_format(self):
        """Covert VIA json annotation format to YOLO format
        Args:
            None
        """
        self.__Converter.via2yolo(output_dir=self.__output_dir)

    def convert_to_yolov5_format(self, class_id):
        """Covert VIA json annotation format to YOLOv5 format
        Args:
            None
        """
        self.__Converter.via2yolov5(class_id, output_dir=self.__output_dir)

    def convert_to_unet_format(self):
        """Covert VIA json annotation format to U-Net format
        Args:
            None
        """
        self.__Converter.via2unet(output_dir=self.__output_dir)

    def convert_to_iccv09_format(self, class_dict):
        """Covert VIA json annotation format to ICCV09 format
        Args:
            class_dict: class dictionary
        """
        self.__Converter.via2iccv09(class_dict, self.__output_dir)

    def convert_to_pascalvoc_format(self): # not yet tested
        """Covert VIA json annotation format to Pascal VOC format
        Args:
            None
        """
        self.__Converter.via2pascalvoc(self.__output_dir)

    def augment(self, aug, aug_engine='imgaug'):
        self.__Augmenter.augment(aug, aug_engine=aug_engine, output_dir=self.__output_dir)

    def transform(self, tf, aug_engine='imgaug', add_name='', numeric_file_name=False):
        self.__Augmenter.transform(aug=tf, aug_engine=aug_engine, output_dir=self.__output_dir, add_name=add_name, numeric_file_name=numeric_file_name)

    def merge(self, dataset_list) -> None:
        out_anns = {}
        out_anns.update(self.__via)
        
        for elm in dataset_list:
            out_anns.update(elm)

        with open(self.__output_dir + "via_region_data.json", "w") as output_file:
            json.dump(out_anns, output_file)

    def numericalSort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def split_train_test(self, class_list, regex_list, train_ratio=0.8):
        
        for regex_idx, regex in enumerate(regex_list):
            file_list = sorted([f for f in os.listdir(self.__input_dir) if re.search(regex, f)], key=DatasetPolygon.numericalSort)

            # create folder for current regex / class
            class_output_dir = self.__output_dir + class_list[regex_idx] + "/"

            # calculate the ratio of test dataset
            test_counter = math.ceil(len(file_list) * (1 - train_ratio))

            # path to original image
            path_to_original_image = self.__input_dir

            path_to_save_test_image = class_output_dir + "val/"
            if not os.path.exists(path_to_save_test_image):
                os.makedirs(path_to_save_test_image)

            path_to_save_train_image = class_output_dir + "train/"
            if not os.path.exists(path_to_save_train_image):
                os.makedirs(path_to_save_train_image)

            # shuffle the file list
            random.shuffle(file_list)
            train_dataset = file_list[test_counter:]
            test_dataset = file_list[:test_counter]

            # untuk mendapatkan filename + key harus menggunakan dataset_dict
            test_via_json = {}
            for file_name in test_dataset:
                file_key = self.__dataset_dict[file_name]['key']
                test_via_json[file_key] = copy.deepcopy(self.__via[file_key])

                image_dst = os.path.join(path_to_save_test_image, file_name)
                image_src = os.path.join(path_to_original_image, file_name)

                # copy original image to destination path
                shutil.copy(image_src, image_dst)

            # generate via json for test dataset
            with open(path_to_save_test_image + "via_region_data.json", "w") as output_file:
                json.dump(test_via_json, output_file)

            train_via_json = {}
            for file_name in train_dataset:
                file_key = self.__dataset_dict[file_name]['key']
                train_via_json[file_key] = copy.deepcopy(self.__via[file_key])

                image_dst = os.path.join(path_to_save_train_image, file_name)
                image_src = os.path.join(path_to_original_image, file_name)

                shutil.copy(image_src, image_dst)

            # generate via json for train dataset
            with open(path_to_save_train_image + "via_region_data.json", "w") as output_file:
                json.dump(train_via_json, output_file)

    # set and get
    def getVIAJSON(self) -> dict:
        return self.__via

    def getDataset(self) -> list:
        return self.__dataset

    def getDatasetDict(self) -> dict:
        return self.__dataset_dict

    def getPolygonData(self) -> dict:
        return self.__Polygon_data

class DatasetBoundingBox:

    # Basic members
    __dataset = None
    __dataset_dict = {}
    __via = None
    length = 0
    __output_dir = None

    # Object class
    class_list = []

    # __Polygon
    __vialibbox = None
    __vialibbox_data = {}

    # Visualizer
    __DatasetVisualizer = None

    # Converter
    __Converter = None

    # Augmenter
    __Augmenter = None
    
    def __init__(self, images_directory, output_directory, annotation_file_name):
        
        json_file = os.path.join(images_directory, annotation_file_name)
        with open(json_file) as f:
            imgs_ann = json.load(f)

        # clean img_anns from non-annotated images
        k_sus = []
        for k, v in imgs_ann.items():
            if len(v['regions']) == 0:
                k_sus.append(k)

        for k in k_sus:
            del imgs_ann[k]

        # creating dataset json
        dataset_list = []
        for idx, v in enumerate(imgs_ann.values()):

            filename = os.path.join(images_directory, v["filename"])

            if cv2.imread(filename) is not None:
                
                height, width = cv2.imread(filename).shape[:2]
                
                record = {}
                record["file_name"] = filename
                record["key"] = v["filename"] + str(v["size"])
                record["image_id"] = idx
                record["height"] = height
                record["width"] = width
            
                annos = v["regions"]
                objs = []

                # only for annotated regions
                if len(annos) > 0:
                    for anno in annos:
                        reg = anno["region_attributes"]
                        anno = anno["shape_attributes"]
                        px = int(anno["x"])
                        py = int(anno["y"])
                        pwidth = int(anno["width"])
                        pheight = int(anno["height"])
                        px2 = int(px + pwidth)
                        py2 = int(py + pheight)
                        region_class = reg["type"]
                        obj = {
                            "bbox": [px, py, px2, py2],
                            "class": region_class,
                            }
                        objs.append(obj)
                    record["annotations"] = objs
                    dataset_list.append(record)
                    self.__dataset_dict[os.path.basename(filename)] = record
            
            # set all data class members
            self.__dataset = dataset_list
            self.__via = imgs_ann
            self.length = len(self.__dataset)
            self.__output_dir = output_directory

            # generate output directory
            if not os.path.exists(self.__output_dir):
                os.mkdir(self.__output_dir)

            # Bounding Box
            self.__vialibbox = Vialiboundingbox(self.__dataset)
            self.__vialibbox_data, self.class_list = self.__vialibbox.get_bboxes()

            # Visualizer
            self.__DatasetVisualizer = VisualizerBoundingBoxes(self.__dataset)

            # Converter
            self.__Converter = ConverterBoundinBox(self.__dataset, self.class_list)

            # Augmenter
            self.__Augmenter = AugmenterBoundingBox(self.__dataset, self.__via, self.__vialibbox_data)

    def printDataset(self):
        """Print dataset correspond to this object
        Args:
            None
        """
        for i in range(len(self.__dataset)):
            print(self.__dataset[i])

    def printClasses(self):
        """Print all classes correspond to this object
        Args:
            None
        """
        print(self.class_list)

    def plot_dataset(self, thickness=5, cols=2, image_start=0, num_of_sample=4, color_dict=None, randomness=True):
        """Plot the dataset
        Args:
            cols: nb. fo columns in plot
            num_of_sample: sample that need to be plotted
            color_dict: color dictionary for each object
            semantic_segmentation: (for semantic segmentation only) set this to true if you in case semantic segmentation
            randomness: set True if you want to display your images randomly
        """
        self.__DatasetVisualizer.plot_dataset(bboxes_data=self.__vialibbox_data, thickness=thickness, cols=cols, image_start=image_start, num_of_sample=num_of_sample, color_dict=color_dict, randomness=randomness)

    ###########################  CONVERTER ############################################
    def convert_to_yolo_format(self):
        """Covert VIA json annotation format to YOLO format
        Args:
            None
        """
        self.__Converter.via2yolo(output_dir=self.__output_dir)

    def convert_to_pascalvoc_format(self): # not yet tested
        """Covert VIA json annotation format to Pascal VOC format
        Args:
            None
        """
        self.__Converter.via2pascalvoc(self.__output_dir)

    def augment(self, aug, aug_engine='imgaug'):
        self.__Augmenter.augment(aug, aug_engine, self.__output_dir)

    def transform(self, tf, aug_engine='imgaug', add_name="", numeric_file_name=False):
        self.__Augmenter.transform(tf, aug_engine, self.__output_dir, add_name, numeric_file_name)

    def split_train_test(self, class_list, regex_list, train_ratio=0.8):
        
        for regex_idx, regex in enumerate(regex_list):
            file_list = sorted([f for f in os.listdir(self.__input_dir) if re.search(regex, f)], key=DatasetPolygon.numericalSort)

            # create folder for current regex / class
            class_output_dir = self.__output_dir + class_list[regex_idx] + "/"

            # calculate the ratio of test dataset
            test_counter = math.ceil(len(file_list) * (1 - train_ratio))

            # path to original image
            path_to_original_image = self.__input_dir

            path_to_save_test_image = class_output_dir + "val/"
            if not os.path.exists(path_to_save_test_image):
                os.makedirs(path_to_save_test_image)

            path_to_save_train_image = class_output_dir + "train/"
            if not os.path.exists(path_to_save_train_image):
                os.makedirs(path_to_save_train_image)

            # shuffle the file list
            random.shuffle(file_list)
            train_dataset = file_list[test_counter:]
            test_dataset = file_list[:test_counter]

            # untuk mendapatkan filename + key harus menggunakan dataset_dict
            test_via_json = {}
            for file_name in test_dataset:
                file_key = self.__dataset_dict[file_name]['key']
                test_via_json[file_key] = copy.deepcopy(self.__via[file_key])

                image_dst = os.path.join(path_to_save_test_image, file_name)
                image_src = os.path.join(path_to_original_image, file_name)

                # copy original image to destination path
                shutil.copy(image_src, image_dst)

            # generate via json for test dataset
            with open(path_to_save_test_image + "via_region_data.json", "w") as output_file:
                json.dump(test_via_json, output_file)

            train_via_json = {}
            for file_name in train_dataset:
                file_key = self.__dataset_dict[file_name]['key']
                train_via_json[file_key] = copy.deepcopy(self.__via[file_key])

                image_dst = os.path.join(path_to_save_train_image, file_name)
                image_src = os.path.join(path_to_original_image, file_name)

                shutil.copy(image_src, image_dst)

            # generate via json for train dataset
            with open(path_to_save_train_image + "via_region_data.json", "w") as output_file:
                json.dump(train_via_json, output_file)

    def merge(self, dataset_list) -> None:
        out_anns = {}
        out_anns.update(self.__via)
        
        for elm in dataset_list:
            out_anns.update(elm)

        with open(self.__output_dir + "via_region_data.json", "w") as output_file:
            json.dump(out_anns, output_file)

    # set and get
    def getVIAJSON(self) -> dict:
        return self.__via

    def getDataset(self) -> list:
        return self.__dataset

    def getDatasetDict(self) -> dict:
        return self.__dataset_dict

    def getBoundingBoxes(self) -> dict:
        return self.__vialibbox_data



        
        