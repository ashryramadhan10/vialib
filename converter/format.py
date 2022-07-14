# Yolo class packages
import os
import numpy as np
import glob
import cv2
import shutil
from . import calculateanchors
from pascal_voc_writer import Writer
import argparse
import sys
from .transformer import Transformer
from imgaug import ia
from tqdm import tqdm
import xml.etree.ElementTree as ET

class ConverterPolygons:

    __dataset = None
    __class_list = None
    __polys_dict = None

    def __init__(self, dataset, class_list, polys_dict):
        self.__dataset = dataset
        self.__class_list = class_list
        self.__polys_dict = polys_dict

    def via2yolo(self, output_dir):
        yoloformat = Yoloformat()
        yoloformat.via2yolo(self.__dataset, self.__class_list, output_dir)

    def via2yolov5(self, class_dict, output_dir):
        yolov5 = Yolov5()
        yolov5.via2yolov5(self.__dataset, class_dict, output_dir)

    def via2unet(self, output_dir):
        unetformat = Unetformat()
        unetformat.via2unet(self.__dataset, self.__polys_dict, output_dir)

    def via2iccv09(self, color_dict, output_dir):
        iccv09format = Iccv09format()
        iccv09format.via2iccv09(self.__dataset, color_dict, self.__polys_dict, output_dir)

    def via2pascalvoc(self, output_dir):
        pascalvocformat = Pascalvocformat()
        pascalvocformat.via2pascalvoc(self.__dataset, output_dir)

class ConverterBoundinBox:

    __dataset = None
    __class_list = None

    def __init__(self, dataset, class_list):
        self.__dataset = dataset
        self.__class_list = class_list

    def via2yolo(self, output_dir):
        yoloformat = Yoloformat()
        yoloformat.via2yolo(self.__dataset, self.__class_list, output_dir)

    def via2pascalvoc(self, output_dir):
        pascalvocformat = Pascalvocformat()
        pascalvocformat.via2pascalvoc(self.__dataset, output_dir)

class Pascalvocformat:

    def __init__(self) -> None:
        pass

    def via2pascalvoc(self, dataset, output_dir):

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        pascal_voc_output_dir = output_dir + "pascal_voc/"

        if not os.path.exists(pascal_voc_output_dir):
             os.mkdir(pascal_voc_output_dir)

        for image_file in range(0, len(dataset)):
            file_path = dataset[image_file]['file_name']
            file_name_only = dataset[image_file]['file_name'].split("/")[1].split(".")[0]

            writer = Writer(file_path, dataset[image_file]['width'], dataset[image_file]['height'])

            for i in range(0, len(dataset[image_file]['annotations'])):
                writer.addObject(dataset[image_file]['annotations'][i]['class'], 
                                dataset[image_file]['annotations'][i]['bbox'][0],
                                dataset[image_file]['annotations'][i]['bbox'][1],
                                dataset[image_file]['annotations'][i]['bbox'][2],
                                dataset[image_file]['annotations'][i]['bbox'][3])

            writer.save(pascal_voc_output_dir + file_name_only + ".xml")

    def read_file(xml_file: str):
        
        tree = ET.parse(xml_file)
        root = tree.getroot()

        list_with_all_boxes = []

        for boxes in root.iter('object'):

            ymin, xmin, ymax, xmax, name = None, None, None, None, None

            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            name = boxes.find("name").text

            list_with_all_boxes.append([xmin, ymin, xmax, ymax, name])
        
        return list_with_all_boxes

    def update_file(xml_file: str, bboxes, sizes, image_filename, output_dir):

        xml_output_dir = output_dir + "pascal_voc/"

        if not os.path.exists(xml_output_dir):
            os.mkdir(xml_output_dir)

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # change filename
        filename = root.find('filename').text
        root.find('filename').text = str("aug_" + filename)

        # change filepath
        splitted_filepath = root.find('path').text.split('\\')
        splitted_filepath[-1] = str("aug_" + filename)
        file_path = '\\'.join(splitted_filepath)
        root.find('path').text = str(file_path)

        for size in root.iter('size'):
            size.find('width').text = str(int(sizes[0]))
            size.find('height').text = str(int(sizes[1]))

        i = 0
        for boxes in root.iter('object'):
            boxes.find("bndbox/ymin").text = str(int(bboxes[i][1]))
            boxes.find("bndbox/xmin").text = str(int(bboxes[i][0]))
            boxes.find("bndbox/ymax").text = str(int(bboxes[i][3]))
            boxes.find("bndbox/xmax").text = str(int(bboxes[i][2]))
            boxes.find("name").text = str(boxes[i][4])
            i += 1

        out_file = xml_output_dir + "aug_" + image_filename + ".xml"
        tree.write(out_file)

class Yoloformat:
    
    def __init__(self) -> None:
        pass

    def via2yolo(self, dataset, class_list, output_dir):

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.__convert_via_to_pascal_voc(dataset, output_dir)
        self.__convert_pascal_voc_to_yolo(class_list, output_dir)

        input_images_type = self.__create_preparation_files(dataset, class_list, output_dir)
        self.__split_train_test_dataset(input_images_type)
        self.__calculate_anchors()

    def __convert_via_to_pascal_voc(self, dataset, output_dir):
        
        pascal_voc_output_dir = output_dir + "pascal_voc/"

        if not os.path.exists(pascal_voc_output_dir):
             os.mkdir(pascal_voc_output_dir)

        for image_file in range(0, len(dataset)):
            file_path = dataset[image_file]['file_name']
            file_name_only = dataset[image_file]['file_name'].split("/")[1].split(".")[0]

            writer = Writer(file_path, dataset[image_file]['width'], dataset[image_file]['height'])

            for i in range(0, len(dataset[image_file]['annotations'])):
                writer.addObject(dataset[image_file]['annotations'][i]['class'], 
                                dataset[image_file]['annotations'][i]['bbox'][0],
                                dataset[image_file]['annotations'][i]['bbox'][1],
                                dataset[image_file]['annotations'][i]['bbox'][2],
                                dataset[image_file]['annotations'][i]['bbox'][3])

            writer.save(pascal_voc_output_dir + file_name_only + ".xml")

    def __convert_pascal_voc_to_yolo(self, class_list, output_dir):

        # set output directory
        pascal_voc_dir = output_dir + "pascal_voc/"
        yolo_format_output_dir = output_dir + "yolo_format/"
        class_file = output_dir + "classes.txt"

        if not os.path.exists(yolo_format_output_dir):
            os.mkdir(yolo_format_output_dir)

        with open(class_file, 'w') as f:
            for item in class_list:
                f.write("%s\n" % item)
            
        xml_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), pascal_voc_dir)
        if not os.path.exists(xml_dir):
            print("Provide the correct folder for xml files.")
            sys.exit()

        out_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), yolo_format_output_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.access(out_dir, os.W_OK):
            print("%s folder is not writeable." % out_dir)
            sys.exit()
        
        class_file = os.path.join(os.path.dirname(os.path.realpath('__file__')), class_file)

        if not os.access(class_file, os.F_OK):
            print("%s file is missing." % class_file)
            sys.exit()

        if not os.access(class_file, os.R_OK):
            print("%s file is not readable." % class_file)
            sys.exit()
        
        transformer = Transformer(xml_dir=xml_dir, out_dir=out_dir, class_file=class_file)
        transformer.transform()

    def __create_preparation_files(self, dataset, class_list, output_dir):
        dataset_json = dataset

        data_dir = 'data/'
        dataset_dir = 'data/images/'
        input_images_type = ''

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            os.mkdir(dataset_dir)

        nb_of_class = len(class_list)

        with open(data_dir + "obj.data", 'w') as f:
            f.write("classes = %s\n" % nb_of_class)
            f.write("train = data/train.txt\n")
            f.write("test = data/test.txt\n")
            f.write("names = data/obj.names\n")
            f.write("backup = backup/")

        with open(data_dir + "obj.names", 'w') as f:
            for item in class_list:
                f.write("%s\n" % item)

        for dataset_json_idx in range(len(dataset_json)):
            for file_path in glob.glob(dataset_json[dataset_json_idx]['file_name']):
                file_name = os.path.basename(file_path)
                file_dir = file_path.split("/")[0]
                input_images_type = file_name.split(".")[1]
        
                # copy original file
                source = file_path
                destination = dataset_dir + file_name

                # copy annotation file
                annotation_source = output_dir + "yolo_format/" + file_name.split(".")[0] + ".txt"
                annotation_destination = dataset_dir + file_name.split(".")[0] + ".txt"

                try:
                    shutil.copy(source, destination)
                    shutil.copy(annotation_source, annotation_destination)
                
                # If source and destination are same
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")
                
                # If there is any permission issue
                except PermissionError:
                    print("Permission denied.")
                
                # For other errors
                except:
                    print("Error occurred while copying file.")

        return "." + input_images_type

    def __split_train_test_dataset(self, input_image_type):
        # Current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        current_dir = 'data/images'

        # Percentage of images to be used for the test set
        percentage_test = 20;

        # Create and/or truncate train.txt and test.txt
        file_train = open('data/train.txt', 'w')
        file_test = open('data/test.txt', 'w')

        # Populate train.txt and test.txt
        counter = 1
        index_test = round(100 / percentage_test)
        for pathAndFilename in glob.iglob(os.path.join(current_dir, "*" + input_image_type)):
            title, ext = os.path.splitext(os.path.basename(pathAndFilename))

            if counter == index_test:
                counter = 1
                file_test.write(current_dir + "/" + title + input_image_type + "\n")
            else:
                file_train.write(current_dir + "/" + title + input_image_type + "\n")
                counter = counter + 1

    def __calculate_anchors(self):
        output_dir = "data/"
        filelist='data/train.txt'
        # num_clusters = len(self.__class_list)
        num_clusters = 6

        calculateanchors.calculate_anchors(filelist, output_dir, num_clusters)

class Unetformat:

    def __init__(self) -> None:
        pass

    def via2unet(self, dataset, polys, output_dir):
        output_mask_dir = output_dir + "masks/"

        if not os.path.exists(output_mask_dir):
            os.mkdir(output_mask_dir)

        for dataset_idx in range(len(dataset)):
            for file_path in glob.glob(dataset[dataset_idx]['file_name']):
                file_name = os.path.basename(file_path)

                img = cv2.imread(file_path)
                background = np.zeros_like(img)

                psoi = ia.PolygonsOnImage(polys[file_name]['polygons'],
                            shape=img.shape)

                mask = psoi.draw_on_image(background, 
                    color_face=[255, 255, 255], 
                    alpha_face=1, alpha_points=0, alpha_lines=0,
                    size_points=0, size_lines=0,
                    color_lines=[255, 255, 255])

                cv2.imwrite(output_mask_dir + file_name, mask)  

class Iccv09format:

    def __init__(self) -> None:
        pass

    def via2iccv09(self, dataset, color_dict, polys, output_dir):
        output_region_dir = output_dir + "labels/"

        if not os.path.exists(output_region_dir):
            os.mkdir(output_region_dir)
        
        counter = 1
        n = len(dataset)

        for dataset_idx in range(len(dataset)):
            for file_path in glob.glob(dataset[dataset_idx]['file_name']):
                file_name = os.path.basename(file_path)
                file_name_only = file_name.split(".")[0]

                img = cv2.imread(file_path)
                mask = np.zeros_like(img)

                psoi = ia.PolygonsOnImage(polys[file_name]['polygons'],
                            shape=img.shape)

                image_polys = mask.copy()
                for psoi_idx in range(len(psoi)):
                    image_polys = psoi[psoi_idx].draw_on_image(image_polys, 
                        alpha_face=1, alpha_points=0, alpha_lines=0,
                        size_points=0, size_lines=0, 
                        color_face=color_dict[polys[file_name]['classes'][psoi_idx]], color_lines=color_dict[polys[file_name]['classes'][psoi_idx]])

                # for unknown area
                regions = mask.astype('int8').copy()

                # for known area
                for idx, (k, v) in enumerate(color_dict.items()):
                    
                    class_pixels = np.where(
                        (image_polys[:, :, 0] == v[0]) &
                        (image_polys[:, :, 1] == v[1]) &
                        (image_polys[:, :, 2] == v[2])
                    )
                    regions[class_pixels] = idx

                np.savetxt(output_region_dir + file_name_only + ".regions.txt", regions[:, :, 0], fmt="%d", delimiter=" ", newline="\n")
            print("Done (" + str(counter) + " / " + str(n) + ") images")
            counter += 1

class Yolov5:

    def __init__(self) -> None:
        pass

    def via2yolov5(self, dataset, class_dict, output_dir):

        # set output directory
        pascal_voc_dir = output_dir.replace("/","\\") + "pascal_voc\\"
        yolo_format_output_dir = output_dir + "yolo_format/"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if not os.path.exists(yolo_format_output_dir):
            os.mkdir(yolo_format_output_dir)
        
        self.__convert_via_to_pascal_voc(dataset, output_dir)

        # Get the annotations
        annotations = [os.path.join(pascal_voc_dir, x) for x in os.listdir(pascal_voc_dir) if x[-3:] == "xml"]
        annotations.sort()

        yolo_format_output_dir = output_dir.replace("/","\\") + "yolo_format\\"

        # Convert and save the annotations
        n = len(annotations)
        for idx, ann in enumerate(tqdm(annotations), start=1):
            info_dict = self.__extract_info_from_xml(ann)
            self.__convert_pascal_voc_to_yolov5(info_dict, class_dict, yolo_format_output_dir)
            print("Done (" + str(idx) + " / " + str(n) + ")")

    def __extract_info_from_xml(self, xml_file):
        root = ET.parse(xml_file).getroot()
    
        # Initialise the info dict 
        info_dict = {}
        info_dict['bboxes'] = []

        # Parse the XML Tree
        for elem in root:
            # Get the file name 
            if elem.tag == "filename":
                info_dict['filename'] = elem.text
                
            # Get the image size
            elif elem.tag == "size":
                image_size = []
                for subelem in elem:
                    image_size.append(int(subelem.text))
                
                info_dict['image_size'] = tuple(image_size)
            
            # Get details of the bounding box 
            elif elem.tag == "object":
                bbox = {}
                for subelem in elem:
                    if subelem.tag == "name":
                        bbox["class"] = subelem.text
                        
                    elif subelem.tag == "bndbox":
                        for subsubelem in subelem:
                            bbox[subsubelem.tag] = int(subsubelem.text)            
                info_dict['bboxes'].append(bbox)
        
        return info_dict

    def __convert_pascal_voc_to_yolov5(self, info_dict, class_name_to_id_mapping, yolo_output_dir):
        print_buffer = []
    
        # For each bounding box
        for b in info_dict["bboxes"]:
            try:
                class_id = class_name_to_id_mapping[b["class"]]
            except KeyError:
                print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
            
            # Transform the bbox co-ordinates as per the format required by YOLO v5
            b_center_x = (b["xmin"] + b["xmax"]) / 2 
            b_center_y = (b["ymin"] + b["ymax"]) / 2
            b_width    = (b["xmax"] - b["xmin"])
            b_height   = (b["ymax"] - b["ymin"])
            
            # Normalise the co-ordinates by the dimensions of the image
            image_w, image_h, image_c = info_dict["image_size"]  
            b_center_x /= image_w 
            b_center_y /= image_h 
            b_width    /= image_w 
            b_height   /= image_h 
            
            #Write the bbox details to the file 
            print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

        # getting image extention
        image_ext = info_dict["filename"][-3:];
            
        # Name of the file which we have to save 
        save_file_name = os.path.join(yolo_output_dir, info_dict["filename"].replace(image_ext, "txt"))
        
        # Save the annotation to disk
        print("\n".join(print_buffer), file = open(save_file_name, "w"))

    def __convert_via_to_pascal_voc(self, dataset, output_dir):
        
        pascal_voc_output_dir = output_dir + "pascal_voc/"

        if not os.path.exists(pascal_voc_output_dir):
             os.mkdir(pascal_voc_output_dir)

        for image_file in range(0, len(dataset)):
            file_path = dataset[image_file]['file_name']
            file_name_only = dataset[image_file]['file_name'].split("/")[1].split(".")[0]

            writer = Writer(file_path, dataset[image_file]['width'], dataset[image_file]['height'])

            for i in range(0, len(dataset[image_file]['annotations'])):
                writer.addObject(dataset[image_file]['annotations'][i]['class'], 
                                dataset[image_file]['annotations'][i]['bbox'][0],
                                dataset[image_file]['annotations'][i]['bbox'][1],
                                dataset[image_file]['annotations'][i]['bbox'][2],
                                dataset[image_file]['annotations'][i]['bbox'][3])

            writer.save(pascal_voc_output_dir + file_name_only + ".xml")
            
        

