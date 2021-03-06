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

class AugmenterPolygon:

    __dataset = None
    __via = None
    __polys = None

    def __init__(self, dataset, via, polys):
        self.__dataset = dataset
        self.__via = via
        self.__polys = polys

    def augment(self, aug, output_dir):
        psoi_aug_list = []
        aug_via_json = {}

        for dataset_json_idx in range(len(self.__dataset)):
            for file_idx, file_path in enumerate(glob.glob(self.__dataset[dataset_json_idx]['file_name'])):
                file_name = os.path.basename(file_path)
                file_dir = file_path.split("/")[0]

                img = cv2.imread(file_path)

                psoi = ia.PolygonsOnImage(self.__polys[file_name]['polygons'],
                            shape=img.shape)

                image_aug, psoi_aug = aug(image=img, polygons=psoi)

                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                cv2.imwrite(output_dir + "aug_" + file_name, image_aug)

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

                # add psoi aug to list
                psoi_aug_list.append(psoi_aug)

        # more robust to missing images
        for imgs_idx in range(len(self.__dataset)):
            key = self.__dataset[imgs_idx]["key"]
            aug_via_json_key = "aug_" + key
            aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[key])
            aug_via_json[aug_via_json_key]['filename'] = "aug_" + aug_via_json[aug_via_json_key]['filename']
            
            annos = aug_via_json[aug_via_json_key]["regions"]
            if len(annos) > 0:
                for anno_idx, anno in enumerate(annos):
                    anno = anno["shape_attributes"]
                    anno["all_points_x"] = [l.tolist() for l in psoi_aug_list[imgs_idx][anno_idx].exterior[:,0].astype(int)]
                    anno["all_points_y"] = [l.tolist() for l in psoi_aug_list[imgs_idx][anno_idx].exterior[:,1].astype(int)]

        # for imgs_idx, (k, v) in enumerate(self.__via.items()):
        #     aug_via_json_key = "aug_" + k
        #     aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[k])

        out_anns = {}
        for d in [self.__via, aug_via_json]:
            out_anns.update(d)

        with open(output_dir + "via_region_data.json", "w") as output_file:
            json.dump(out_anns, output_file)

    def transform(self, aug, output_dir, add_name="", numeric_file_name=False):
        psoi_aug_list = []
        aug_via_json = {}

        for dataset_json_idx in range(len(self.__dataset)):
            for file_idx, file_path in enumerate(glob.glob(self.__dataset[dataset_json_idx]['file_name'])):
                file_name = os.path.basename(file_path)
                file_dir = file_path.split("/")[0]

                img = cv2.imread(file_path)

                psoi = ia.PolygonsOnImage(self.__polys[file_name]['polygons'],
                            shape=img.shape)

                image_aug, psoi_aug = aug(image=img, polygons=psoi)

                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                if numeric_file_name:
                    cv2.imwrite(output_dir + add_name + str(dataset_json_idx) + "." + file_name.split(".")[1], image_aug)
                else:
                    cv2.imwrite(output_dir + file_name, image_aug)

                # add psoi aug to list
                psoi_aug_list.append(psoi_aug)

        # more robust to missing images
        for imgs_idx in range(len(self.__dataset)):
            key = self.__dataset[imgs_idx]["key"]

            if numeric_file_name:
                via_filename = self.__via[key]['filename']
                via_size = self.__via[key]['size']
                aug_via_json_key = add_name + str(imgs_idx) + "." + via_filename.split(".")[1] + str(via_size)
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[key])
                aug_via_json[aug_via_json_key]['filename'] = add_name + str(imgs_idx) + "." + aug_via_json[aug_via_json_key]['filename'].split(".")[1]
            else:
                aug_via_json_key = key
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[key])
                aug_via_json[aug_via_json_key]['filename'] = aug_via_json[aug_via_json_key]['filename']
            
            annos = aug_via_json[aug_via_json_key]["regions"]
            if len(annos) > 0:
                for anno_idx, anno in enumerate(annos):
                    anno = anno["shape_attributes"]
                    anno["all_points_x"] = [l.tolist() for l in psoi_aug_list[imgs_idx][anno_idx].exterior[:,0].astype(int)]
                    anno["all_points_y"] = [l.tolist() for l in psoi_aug_list[imgs_idx][anno_idx].exterior[:,1].astype(int)]

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

    def augment(self, aug, output_dir):
        bboxes_aug_list = []
        aug_via_json = {}

        for dataset_json_idx in range(len(self.__dataset)):
            for file_idx, file_path in enumerate(glob.glob(self.__dataset[dataset_json_idx]['file_name'])):
                file_name = os.path.basename(file_path)
                file_dir = file_path.split("/")[0]

                img = cv2.imread(file_path)

                bbsoi = ia.BoundingBoxesOnImage(self.__bboxes[file_name]['bboxes'],
                            shape=img.shape)

                image_aug, bbsoi_aug = aug(image=img, bounding_boxes=bbsoi)

                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                cv2.imwrite(output_dir + "aug_" + file_name, image_aug)

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

                # add psoi aug to list
                bboxes_aug_list.append(bbsoi_aug)

        # more robust to missing images
        for imgs_idx in range(len(self.__dataset)):
            key = self.__dataset[imgs_idx]["key"]
            aug_via_json_key = "aug_" + key
            aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[key])
            aug_via_json[aug_via_json_key]['filename'] = "aug_" + aug_via_json[aug_via_json_key]['filename']
            
            annos = aug_via_json[aug_via_json_key]["regions"]
            if len(annos) > 0:
                for anno_idx, anno in enumerate(annos):
                    anno = anno["shape_attributes"]
                    anno["x"] = int(bboxes_aug_list[imgs_idx][anno_idx].x1)
                    anno["y"] = int(bboxes_aug_list[imgs_idx][anno_idx].y1)
                    anno["width"] = int(bboxes_aug_list[imgs_idx][anno_idx].x2 - bboxes_aug_list[imgs_idx][anno_idx].x1)
                    anno["height"] = int(bboxes_aug_list[imgs_idx][anno_idx].y2 - bboxes_aug_list[imgs_idx][anno_idx].y1)

        out_anns = {}
        for d in [self.__via, aug_via_json]:
            out_anns.update(d)

        with open(output_dir + "via_region_data.json", "w") as output_file:
            json.dump(out_anns, output_file)

    def transform(self, aug, output_dir, add_name="", numeric_file_name=False):
        bboxes_aug_list = []
        aug_via_json = {}

        for dataset_json_idx in range(len(self.__dataset)):
            for file_idx, file_path in enumerate(glob.glob(self.__dataset[dataset_json_idx]['file_name'])):
                file_name = os.path.basename(file_path)
                file_dir = file_path.split("/")[0]

                img = cv2.imread(file_path)

                bbsoi = ia.BoundingBoxesOnImage(self.__bboxes[file_name]['bboxes'],
                            shape=img.shape)

                image_aug, bbsoi_aug = aug(image=img, bounding_boxes=bbsoi)

                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                if numeric_file_name:
                    cv2.imwrite(output_dir + add_name + str(dataset_json_idx) + "." + file_name.split(".")[1], image_aug)
                else:
                    cv2.imwrite(output_dir + file_name, image_aug)


                # add psoi aug to list
                bboxes_aug_list.append(bbsoi_aug)

        for imgs_idx in range(len(self.__dataset)):
            key = self.__dataset[imgs_idx]["key"]

            if numeric_file_name:
                via_filename = self.__via[key]['filename']
                via_size = self.__via[key]['size']
                aug_via_json_key = add_name + str(imgs_idx) + "." + via_filename.split(".")[1] + str(via_size)
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[key])
                aug_via_json[aug_via_json_key]['filename'] = add_name + str(imgs_idx) + "." + aug_via_json[aug_via_json_key]['filename'].split(".")[1]
            else:
                aug_via_json_key = key
                aug_via_json[aug_via_json_key] = copy.deepcopy(self.__via[key])
                aug_via_json[aug_via_json_key]['filename'] = aug_via_json[aug_via_json_key]['filename']
            
            annos = aug_via_json[aug_via_json_key]["regions"]
            if len(annos) > 0:
                for anno_idx, anno in enumerate(annos):
                    anno = anno["shape_attributes"]
                    anno["x"] = int(bboxes_aug_list[imgs_idx][anno_idx].x1)
                    anno["y"] = int(bboxes_aug_list[imgs_idx][anno_idx].y1)
                    anno["width"] = int(bboxes_aug_list[imgs_idx][anno_idx].x2 - bboxes_aug_list[imgs_idx][anno_idx].x1)
                    anno["height"] = int(bboxes_aug_list[imgs_idx][anno_idx].y2 - bboxes_aug_list[imgs_idx][anno_idx].y1)

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
