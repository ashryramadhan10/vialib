import os
import json

import numpy as np
import cv2

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
    __via = None
    length = 0
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

            self.__dataset = dataset_list
            self.__via = imgs_ann
            self.length = len(self.__dataset)
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

    def plot_dataset(self, cols=2, image_start=0, num_of_sample=4, color_dict=None, semantic_segmentation=False, randomness=True):
        """Plot the dataset
        Args:
            cols: nb. fo columns in plot
            num_of_sample: sample that need to be plotted
            color_dict: color dictionary for each object
            semantic_segmentation: (for semantic segmentation only) set this to true if you in case semantic segmentation
            randomness: set True if you want to display your images randomly
        """
        self.__DatasetVisualizer.plot_dataset(polygon_data=self.__Polygon_data, cols=cols, image_start=image_start, num_of_sample=num_of_sample, color_dict=color_dict, semantic_segmentation=semantic_segmentation, randomness=randomness)

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

    def augment(self, aug):
        self.__Augmenter.augment(aug, self.__output_dir)

    def transform(self, tf, add_name="", numeric_file_name=False):
        self.__Augmenter.transform(tf, self.__output_dir, add_name, numeric_file_name)

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

    def getPolygonData(self) -> dict:
        return self.__Polygon_data

class DatasetBoundingBox:

    # Basic members
    __dataset = None
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

    def plot_dataset(self, cols=2, image_start=0, num_of_sample=4, color_dict=None, randomness=True):
        """Plot the dataset
        Args:
            cols: nb. fo columns in plot
            num_of_sample: sample that need to be plotted
            color_dict: color dictionary for each object
            semantic_segmentation: (for semantic segmentation only) set this to true if you in case semantic segmentation
            randomness: set True if you want to display your images randomly
        """
        self.__DatasetVisualizer.plot_dataset(bboxes_data=self.__vialibbox_data, cols=cols, image_start=image_start, num_of_sample=num_of_sample, color_dict=color_dict, randomness=randomness)

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

    def augment(self, aug):
        self.__Augmenter.augment(aug, self.__output_dir)

    def transform(self, tf, add_name="", numeric_file_name=False):
        self.__Augmenter.transform(tf, self.__output_dir, add_name, numeric_file_name)

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

    def getBoundingBoxes(self) -> dict:
        return self.__vialibbox_data



        
        