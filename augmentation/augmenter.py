import imgaug as ia
import os
import glob
import cv2
import json
import copy
import shutil
import re
import matplotlib.pyplot as plt
from vialib.converter.format import Pascalvocformat
import numpy as np
import random
import imgaug as ia

class AugmenterPolygon:

    __dataset = None
    __via = None
    __polys = None

    def __init__(self, dataset, via, polys):
        self.__dataset = dataset
        self.__via = via
        self.__polys = polys

    def augment(self, aug, aug_engine, output_dir, repeat):

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if aug_engine == 'imgaug':
            all_transformed = {}
            aug_via_json = {}

            for dataset in self.__dataset:
                file_path = dataset['file_name']
                file_name = os.path.basename(file_path)

                img = cv2.imread(file_path)

                psoi = ia.PolygonsOnImage(self.__polys[file_name]['polygons'],
                            shape=img.shape)

                for repeat_idx in range(0, repeat):
                    seed_num = random.randint(0, 100)
                    ia.seed(seed_num)
                    image_aug, psoi_aug = aug(image=img, polygons=psoi)

                    if len(psoi_aug) > 0:
                        all_transformed["aug_" + str(repeat_idx).zfill(2) + "_" + file_name] = {
                            'key': dataset['key'],
                            'psoi': psoi_aug,
                            'file_name': "aug_" + str(repeat_idx).zfill(2) + "_" + file_name,
                        }
                        cv2.imwrite(output_dir + "aug_" + str(repeat_idx).zfill(2) + "_" + file_name, image_aug)

                # copy original file
                source = file_path
                destination = output_dir + file_name
                try:
                    shutil.copy(source, destination)
                
                # If source and destination are same
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")
                
                # If there is any permission issue
                except PermissionError:
                    print("Permission denied.")
                
                # For other errors
                except:
                    print("Error occurred while copying file.")

            # more robust to missing images because based on self.__dataset not self.__via
            for k, v in all_transformed.items():
                aug_via_json_key = v['file_name'] + str(self.__via[v['key']]['size'])
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[v['key']])
                aug_via_json[aug_via_json_key]['filename'] = v['file_name']

                annos = aug_via_json[aug_via_json_key]["regions"]
                if len(annos) > 0:
                    for anno_idx, anno in enumerate(annos):
                        anno = anno["shape_attributes"]
                        anno["all_points_x"] = [l.tolist() for l in v['psoi'][anno_idx].exterior[:,0].astype(int)]
                        anno["all_points_y"] = [l.tolist() for l in v['psoi'][anno_idx].exterior[:,1].astype(int)]

            out_anns = {}
            for d in [self.__via, aug_via_json]:
                out_anns.update(d)

            with open(output_dir + "via_region_data.json", "w") as output_file:
                json.dump(out_anns, output_file)

        elif aug_engine == 'albumentations':
            
            aug_via_json = {}
            all_transformed = {}

            for dataset in self.__dataset:
                file_path = dataset['file_name']
                file_name = os.path.basename(file_path)

                image = cv2.imread(file_path)
                polygons = self.__polys[file_name]['polygons']

                for repeat_idx in range(0, repeat):

                    # seperate the joined keypoints (image_keypoints)
                    transformed_keypoints_list = []
                    seed_num = random.randint(0, 100)

                    for polygon in polygons:
                        transposed_polygon = polygon.coords.T.astype(np.uint16)
                        transposed_polygon[0] = transposed_polygon[0].clip(max=np.max(transposed_polygon[0])-1)
                        transposed_polygon[1] = transposed_polygon[1].clip(max=np.max(transposed_polygon[1])-1)
                        keypoints = list(zip(transposed_polygon[0], transposed_polygon[1]))

                        # augment the image
                        random.seed(seed_num)
                        transformed = aug(image=image, keypoints=keypoints)
                        transformed_image = transformed['image']
                        transformed_keypoints = transformed['keypoints']

                        # if there is any keypoints on the image
                        if len(transformed_keypoints) > 0:
                            transformed_keypoints_list.append(transformed_keypoints)

                    # append to global all_transformed_keypoints list
                    if len(transformed_keypoints_list) > 0:
                        all_transformed["aug_" + str(repeat_idx).zfill(2) + "_" + file_name] = {
                            'key': dataset['key'],
                            'keypoints': transformed_keypoints_list,
                            'file_name': "aug_" + str(repeat_idx).zfill(2) + "_" + file_name,
                            'classes': self.__polys[file_name]['classes'],
                        }

                        cv2.imwrite(output_dir + "aug_" + str(repeat_idx).zfill(2) + "_" + file_name, transformed_image)

                # copy original file
                source = file_path
                destination = output_dir + file_name
                try:
                    shutil.copy(source, destination)
                
                # If source and destination are same
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")
                
                # If there is any permission issue
                except PermissionError:
                    print("Permission denied.")
                
                # For other errors
                except:
                    print("Error occurred while copying file.") 

            for k, v in all_transformed.items():
                aug_via_json_key = v['file_name'] + str(self.__via[v['key']]['size'])
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[v['key']])
                aug_via_json[aug_via_json_key]['filename'] = v['file_name']

                del aug_via_json[aug_via_json_key]['regions']

                regions = []
                for k_idx, keypoints in enumerate(v['keypoints']):
                    regions_data = {
                        'shape_attributes': {
                            'name': 'polygon',
                            'all_points_x': np.array(keypoints)[:, 0].astype(np.uint16).tolist(),
                            'all_points_y': np.array(keypoints)[:, 1].astype(np.uint16).tolist(),
                        },
                        'region_attributes': {
                            'type': v['classes'][k_idx]
                        }
                    }
                    regions.append(regions_data)
                aug_via_json[aug_via_json_key]["regions"] = regions

            out_anns = {}
            for d in [self.__via, aug_via_json]:
                out_anns.update(d)

            with open(output_dir + "via_region_data.json", "w") as output_file:
                json.dump(out_anns, output_file)

    def transform(self, aug, aug_engine, output_dir, add_name, numeric_file_name, repeat):

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if aug_engine == 'imgaug':
            all_transformed = {}
            aug_via_json = {}

            for dataset_idx, dataset in enumerate(self.__dataset):
                file_path = dataset['file_name']
                file_name = os.path.basename(file_path)

                img = cv2.imread(file_path)

                psoi = ia.PolygonsOnImage(self.__polys[file_name]['polygons'],
                            shape=img.shape)

                for repeat_idx in range(0, repeat):
                    seed_num = random.randint(0, 100)
                    ia.seed(seed_num)
                    image_aug, psoi_aug = aug(image=img, polygons=psoi)

                    if len(psoi_aug) > 0:
                        if numeric_file_name:
                            all_transformed[add_name + str(dataset_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1]] = {
                                'key': dataset['key'],
                                'psoi': psoi_aug,
                                'file_name': add_name + str(dataset_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1],
                            }
                            cv2.imwrite(output_dir + add_name + str(dataset_idx) + "." + file_name.split(".")[1], image_aug)
                        else:
                            all_transformed[add_name + str(repeat_idx).zfill(2) + "_" + file_name] = {
                                'key': dataset['key'],
                                'psoi': psoi_aug,
                                'file_name': add_name + str(repeat_idx).zfill(2) + "_" + file_name,
                            }
                            cv2.imwrite(output_dir + add_name + str(repeat_idx).zfill(2) + "_" + file_name, image_aug)
            
            # more robust to missing images because based on self.__dataset not self.__via
            for k, v in all_transformed.items():
                aug_via_json_key = v['file_name'] + str(self.__via[v['key']]['size'])
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[v['key']])
                aug_via_json[aug_via_json_key]['filename'] = v['file_name']

                annos = aug_via_json[aug_via_json_key]["regions"]
                if len(annos) > 0:
                    for anno_idx, anno in enumerate(annos):
                        anno = anno["shape_attributes"]
                        anno["all_points_x"] = [l.tolist() for l in v['psoi'][anno_idx].exterior[:,0].astype(int)]
                        anno["all_points_y"] = [l.tolist() for l in v['psoi'][anno_idx].exterior[:,1].astype(int)]

            with open(output_dir + "via_region_data.json", "w") as output_file:
                json.dump(aug_via_json, output_file)

        elif aug_engine == 'albumentations':
            aug_via_json = {}
            all_transformed = {}

            for dataset_idx, dataset in enumerate(self.__dataset):
                
                file_path = dataset['file_name']
                file_name = os.path.basename(file_path)

                image = cv2.imread(file_path)

                polygons = self.__polys[file_name]['polygons']

                for repeat_idx in range(0, repeat):
                    # seperate the joined keypoints (image_keypoints)
                    transformed_keypoints_list = []
                    seed_num = random.randint(0, 100)

                    for polygon in polygons:
                        transposed_polygon = polygon.coords.T.astype(np.uint16)
                        transposed_polygon[0] = transposed_polygon[0].clip(max=np.max(transposed_polygon[0])-1)
                        transposed_polygon[1] = transposed_polygon[1].clip(max=np.max(transposed_polygon[1])-1)
                        keypoints = list(zip(transposed_polygon[0], transposed_polygon[1]))

                        # augment the image
                        random.seed(seed_num)
                        transformed = aug(image=image, keypoints=keypoints)
                        transformed_image = transformed['image']
                        transformed_keypoints = transformed['keypoints']

                        # you can fine tuning this parameters
                        if len(transformed_keypoints) > 0:
                            transformed_keypoints_list.append(transformed_keypoints)

                    # append to global all_transformed_keypoints list
                    if len(transformed_keypoints_list) > 0:
                        if numeric_file_name:
                            all_transformed[add_name + str(dataset_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1]] = {
                                'key': dataset['key'],
                                'keypoints': transformed_keypoints_list,
                                'file_name': add_name + str(dataset_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1],
                                'classes': self.__polys[file_name]['classes'],
                            }

                            cv2.imwrite(output_dir + add_name + str(dataset_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1], transformed_image)
                        else:
                            all_transformed[add_name + str(repeat_idx).zfill(2) + "_" + file_name] = {
                                'key': dataset['key'],
                                'keypoints': transformed_keypoints_list,
                                'file_name': add_name + str(repeat_idx).zfill(2) + "_" + file_name,
                                'classes': self.__polys[file_name]['classes'],
                            }

                            cv2.imwrite(output_dir + add_name + str(repeat_idx).zfill(2) + "_" + file_name, transformed_image)

            for k, v in all_transformed.items():
                aug_via_json_key = v['file_name'] + str(self.__via[v['key']]['size'])
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[v['key']])
                aug_via_json[aug_via_json_key]['filename'] = v['file_name']

                del aug_via_json[aug_via_json_key]['regions']

                regions = []
                for k_idx, keypoints in enumerate(v['keypoints']):
                    regions_data = {
                        'shape_attributes': {
                            'name': 'polygon',
                            'all_points_x': np.array(keypoints)[:, 0].astype(np.uint16).tolist(),
                            'all_points_y': np.array(keypoints)[:, 1].astype(np.uint16).tolist(),
                        },
                        'region_attributes': {
                            'type': v['classes'][k_idx]
                        }
                    }
                    regions.append(regions_data)
                aug_via_json[aug_via_json_key]["regions"] = regions

            with open(output_dir + "via_region_data.json", "w") as output_file:
                json.dump(aug_via_json, output_file)

class AugmenterBoundingBox:

    __dataset = None
    __via = None
    __bboxes = None

    def __init__(self, dataset, via, bboxes):
        self.__dataset = dataset
        self.__via = via
        self.__bboxes = bboxes

    def augment(self, aug, aug_engine, output_dir, repeat):

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if aug_engine == 'imgaug':
            all_transformed = {}
            aug_via_json = {}

            for dataset in self.__dataset:
                file_path = dataset['file_name']
                file_name = os.path.basename(file_path)

                img = cv2.imread(file_path)

                bbsoi = ia.BoundingBoxesOnImage(self.__bboxes[file_name]['bboxes'],
                            shape=img.shape)
                
                for repeat_idx in range(0, repeat):
                    seed_num = random.randint(0, 100)
                    ia.seed(seed_num)
                    image_aug, bbsoi_aug = aug(image=img, bounding_boxes=bbsoi)

                    if len(bbsoi_aug) > 0:
                        all_transformed["aug_" + str(repeat_idx).zfill(2) + "_" + file_name] = {
                            'key': dataset['key'],
                            'bbsoi': bbsoi_aug,
                            'file_name': "aug_" + str(repeat_idx).zfill(2) + "_" + file_name,
                        }

                    cv2.imwrite(output_dir + "aug_" + str(repeat_idx).zfill(2) + "_" + file_name, image_aug)

                # copy original file
                source = file_path
                destination = output_dir + file_name
                try:
                    shutil.copy(source, destination)
                
                # If source and destination are same
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")
                
                # If there is any permission issue
                except PermissionError:
                    print("Permission denied.")
                
                # For other errors
                except:
                    print("Error occurred while copying file.")

            for k, v in all_transformed.items():
                aug_via_json_key = v['file_name'] + str(self.__via[v['key']]['size'])
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[v['key']])
                aug_via_json[aug_via_json_key]['filename'] = v['file_name']

                annos = aug_via_json[aug_via_json_key]["regions"]
                if len(annos) > 0:
                    for anno_idx, anno in enumerate(annos):
                        anno = anno["shape_attributes"]
                        anno["x"] = int(v['bbsoi'][anno_idx].x1)
                        anno["y"] = int(v['bbsoi'][anno_idx].y1)
                        anno["width"] = int(v['bbsoi'][anno_idx].x2 - v['bbsoi'][anno_idx].x1)
                        anno["height"] = int(v['bbsoi'][anno_idx].y2 - v['bbsoi'][anno_idx].y1)

            out_anns = {}
            for d in [self.__via, aug_via_json]:
                out_anns.update(d)

            with open(output_dir + "via_region_data.json", "w") as output_file:
                json.dump(out_anns, output_file)

        elif aug_engine == 'albumentations':
            all_transformed = {}
            aug_via_json = {}

            for dataset in self.__dataset:
                
                file_path = dataset['file_name']
                file_name = os.path.basename(file_path)

                image = cv2.imread(file_path)
                annos = dataset['annotations']

                for repeat_idx in range(0, repeat):
                    bboxes = []
                    seed_num = random.randint(0, 100)

                    for anno in annos:
                        bbox = [anno['bbox'][0], anno['bbox'][1], anno['bbox'][2], anno['bbox'][3], anno['class']]
                        bboxes.append(bbox)

                    random.seed(seed_num)
                    transformed = aug(image=image, bboxes=bboxes)
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']

                    if len(transformed_bboxes) > 0:
                        all_transformed['aug_' + str(repeat_idx).zfill(2) + '_' + file_name] = {
                            'key': dataset['key'],
                            'bboxes': transformed_bboxes,
                            'file_name': 'aug_' + str(repeat_idx).zfill(2) + '_' + file_name,
                        }

                        cv2.imwrite(output_dir + 'aug_' + str(repeat_idx).zfill(2) + '_' + file_name, transformed_image)

                # copy original file
                source = file_path
                destination = output_dir + file_name
                try:
                    shutil.copy(source, destination)
                
                # If source and destination are same
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")
                
                # If there is any permission issue
                except PermissionError:
                    print("Permission denied.")
                
                # For other errors
                except:
                    print("Error occurred while copying file.")

            for k, v in all_transformed.items():
                aug_via_json_key = v['file_name'] + str(self.__via[v['key']]['size'])
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[v['key']])
                aug_via_json[aug_via_json_key]['filename'] = v['file_name']

                del aug_via_json[aug_via_json_key]['regions']

                regions = []
                for transformed_bboxes in v['bboxes']:
                    regions_data = {
                        'shape_attributes': {
                            'name': 'rect',
                            'x': int(transformed_bboxes[0]),
                            'y': int(transformed_bboxes[1]),
                            'width': int(transformed_bboxes[2] - transformed_bboxes[0]),
                            'height': int(transformed_bboxes[3] - transformed_bboxes[1]),
                        },
                        'region_attributes': {
                            'type': transformed_bboxes[4]
                        }
                    }
                    regions.append(regions_data)
                aug_via_json[aug_via_json_key]["regions"] = regions

            out_anns = {}
            for d in [self.__via, aug_via_json]:
                out_anns.update(d)

            with open(output_dir + "via_region_data.json", "w") as output_file:
                json.dump(out_anns, output_file)
                    
    def transform(self, aug, aug_engine, output_dir, add_name, numeric_file_name, repeat):
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if aug_engine == 'imgaug':
            all_transformed = {}
            aug_via_json = {}

            for dataset_json_idx in range(len(self.__dataset)):
                file_path = self.__dataset[dataset_json_idx]['file_name']
                file_name = os.path.basename(file_path)
                file_dir = file_path.split("/")[0]

                img = cv2.imread(file_path)

                bbsoi = ia.BoundingBoxesOnImage(self.__bboxes[file_name]['bboxes'],
                            shape=img.shape)

                for repeat_idx in range(0, repeat):
                    seed_num = random.randint(0, 100)
                    ia.seed(seed_num)
                    image_aug, bbsoi_aug = aug(image=img, bounding_boxes=bbsoi)

                    if len(bbsoi_aug) > 0:
                        if numeric_file_name:
                            all_transformed[output_dir + add_name + str(dataset_json_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1]] = {
                                'key': dataset['key'],
                                'bbsoi': bbsoi_aug,
                                'file_name': output_dir + add_name + str(dataset_json_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1],
                            }
                            cv2.imwrite(output_dir + add_name + str(dataset_json_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1], image_aug)
                        else:
                            all_transformed[output_dir + add_name + str(repeat_idx).zfill(2) + "_" + file_name] = {
                                'key': dataset['key'],
                                'bbsoi': bbsoi_aug,
                                'file_name': output_dir + add_name + str(repeat_idx).zfill(2) + "_" + file_name,
                            }
                            cv2.imwrite(output_dir + add_name + str(repeat_idx).zfill(2) + "_" + file_name, image_aug)

            for k, v in all_transformed.items():
                aug_via_json_key = v['file_name'] + str(self.__via[v['key']]['size'])
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[v['key']])
                aug_via_json[aug_via_json_key]['filename'] = v['file_name']

                annos = aug_via_json[aug_via_json_key]["regions"]
                if len(annos) > 0:
                    for anno_idx, anno in enumerate(annos):
                        anno = anno["shape_attributes"]
                        anno["x"] = int(v['bbsoi'][anno_idx].x1)
                        anno["y"] = int(v['bbsoi'][anno_idx].y1)
                        anno["width"] = int(v['bbsoi'][anno_idx].x2 - v['bbsoi'][anno_idx].x1)
                        anno["height"] = int(v['bbsoi'][anno_idx].y2 - v['bbsoi'][anno_idx].y1)

            with open(output_dir + "via_region_data.json", "w") as output_file:
                json.dump(aug_via_json, output_file)

        elif aug_engine == 'albumentations':
            all_transformed = {}
            aug_via_json = {}

            for dataset in self.__dataset:
                seed_num = random.randint(0, 100)
                file_path = dataset['file_name']
                file_name = os.path.basename(file_path)

                image = cv2.imread(file_path)
                annos = dataset['annotations']

                for repeat_idx in range(0, repeat):
                    bboxes = []
                    seed_num = random.randint(0, 100)

                    for anno in annos:
                        bbox = [anno['bbox'][0], anno['bbox'][1], anno['bbox'][2], anno['bbox'][3], anno['class']]
                        bboxes.append(bbox)

                    random.seed(seed_num)
                    transformed = aug(image=image, bboxes=bboxes)
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']

                    if len(transformed_bboxes) > 0:

                        if numeric_file_name:
                            all_transformed[add_name + str(dataset_json_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1]] = {
                                'key': dataset['key'],
                                'bboxes': transformed_bboxes,
                                'file_name': add_name + str(dataset_json_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1],
                            }
                            cv2.imwrite(output_dir + add_name + str(dataset_json_idx).zfill(2) + "_" + str(repeat_idx).zfill(2) + "." + file_name.split(".")[1], transformed_image)
                        else:
                            all_transformed[add_name + str(repeat_idx).zfill(2) + "_" + file_name] = {
                                'key': dataset['key'],
                                'bboxes': transformed_bboxes,
                                'file_name': add_name + str(repeat_idx).zfill(2) + "_" + file_name,
                            }
                            cv2.imwrite(output_dir + add_name + str(repeat_idx).zfill(2) + "_" + file_name, transformed_image)
            
            for k, v in all_transformed.items():
                aug_via_json_key = v['file_name'] + str(self.__via[v['key']]['size'])
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[v['key']])
                aug_via_json[aug_via_json_key]['filename'] = v['file_name']

                del aug_via_json[aug_via_json_key]['regions']

                regions = []
                for transformed_bboxes in v['bboxes']:
                    regions_data = {
                        'shape_attributes': {
                            'name': 'rect',
                            'x': int(transformed_bboxes[0]),
                            'y': int(transformed_bboxes[1]),
                            'width': int(transformed_bboxes[2] - transformed_bboxes[0]),
                            'height': int(transformed_bboxes[3] - transformed_bboxes[1]),
                        },
                        'region_attributes': {
                            'type': transformed_bboxes[4]
                        }
                    }
                    regions.append(regions_data)
                aug_via_json[aug_via_json_key]["regions"] = regions

            with open(output_dir + "via_region_data.json", "w") as output_file:
                json.dump(aug_via_json, output_file)

class AugmenterPascalVOC:

    def __init__(self) -> None:
        pass

    def numericalSort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    # using Albumentations
    def augment_pascalvoc_format(aug, input_dir, output_dir): # not tested yet
        
        counter = 1
        n = len(glob.glob(input_dir))

        for filepath in sorted(glob.glob(input_dir), key=AugmenterPascalVOC.numericalSort):
            filename = os.path.basename(filepath).split(".")[0]
            file_ext = os.path.basename(filepath).split(".")[1]

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            xml_f = 'pascal_voc/' + filename + '.xml'
            bboxes = Pascalvocformat.read_file(xml_file=xml_f)

            transformed = aug(image=image, bboxes=bboxes)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']

            sizes = [transformed_image.shape[1], transformed_image.shape[0]]

            # save image and update xml file
            Pascalvocformat.update_file(xml_file=xml_f, bboxes=transformed_bboxes, sizes=sizes, image_filename=filename, output_dir=output_dir)
            plt.imsave('aug_images/aug_' + filename + "." + file_ext, transformed_image)

            print('Done (' + str(counter) + '/' + str(n) + ')')
            counter += 1
