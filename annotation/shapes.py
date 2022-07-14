from imgaug.augmentables.polys import Polygon
from imgaug.augmentables.bbs import BoundingBox

class Vialibpolygon:

    __dataset = None

    def __init__(self, dataset):
        self.__dataset = dataset

    def get_polygons(self):
        """To get all polygon data from the annotation files
        Args:
            output: "dict" or "list"
        """
        polygons = {}
        class_list = []

        for i in range(len(self.__dataset)):
            polygons_temp = []
            classes_temp = []
            bboxes_temp = []
            for j in range(len(self.__dataset[i]["annotations"])):
                polygons_temp.append(Polygon(list(zip(self.__dataset[i]['annotations'][j]['all_points_x'], self.__dataset[i]['annotations'][j]["all_points_y"]))))
                bboxes_temp.append(BoundingBox(self.__dataset[i]['annotations'][j]['bbox'][0], 
                                        self.__dataset[i]['annotations'][j]['bbox'][1], 
                                        self.__dataset[i]['annotations'][j]['bbox'][2], 
                                        self.__dataset[i]['annotations'][j]['bbox'][3]))
                classes_temp.append(self.__dataset[i]['annotations'][j]['class'])
                class_list.append(self.__dataset[i]['annotations'][j]['class'])

            # save all data into dictionary
            polygons[self.__dataset[i]["file_name"].split("/")[1]] = {"polygons": polygons_temp, "bboxes": bboxes_temp, "classes": classes_temp}
        
        # make class distinct
        class_list = sorted(list(set(class_list)))

        return polygons, class_list

class Vialiboundingbox:

    __dataset = None

    def __init__(self, dataset):
        self.__dataset = dataset

    def get_bboxes(self):
        bboxes = {}
        class_list = []

        for i in range(len(self.__dataset)):
            classes_temp = []
            bboxes_temp = []
            for j in range(len(self.__dataset[i]["annotations"])):
                bboxes_temp.append(BoundingBox(self.__dataset[i]['annotations'][j]['bbox'][0], 
                                        self.__dataset[i]['annotations'][j]['bbox'][1], 
                                        self.__dataset[i]['annotations'][j]['bbox'][2], 
                                        self.__dataset[i]['annotations'][j]['bbox'][3]))
                classes_temp.append(self.__dataset[i]['annotations'][j]['class'])
                class_list.append(self.__dataset[i]['annotations'][j]['class'])

            # save all data into dictionary
            bboxes[self.__dataset[i]["file_name"].split("/")[1]] = {"bboxes": bboxes_temp, "classes": classes_temp}
        
        # make class distinct
        class_list = sorted(list(set(class_list)))

        return bboxes, class_list
