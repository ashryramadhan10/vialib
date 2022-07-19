import random
import os

import numpy as np
import glob
import cv2

import imgaug as ia
from imgaug.augmentables.bbs import BoundingBoxesOnImage

class VisualizerPolygons:

    __dataset = None

    def __init__(self, dataset):
        self.__dataset = dataset

    def plot_dataset(self,
        polygon_data,
        cols=2,
        image_start=0,
        num_of_sample=4,
        color_dict=None,
        semantic_segmentation=False,
        randomness=True):
        
        if randomness:
            samples = random.sample(self.__dataset, num_of_sample)
        else:
            samples = self.__dataset[image_start:(image_start + num_of_sample)]

        images_polys = []
        for sample_idx in range(len(samples)):
            for file_idx, file_path in enumerate(glob.glob(samples[sample_idx]['file_name'])):
                file_name = os.path.basename(file_path)
                
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                psoi = ia.PolygonsOnImage(polygon_data[file_name]['polygons'],
                            shape=img.shape)

                bbsoi = BoundingBoxesOnImage(
                    [polygon.to_bounding_box() for polygon in psoi.polygons],
                    shape=psoi.shape
                )
                
                if color_dict == None:
                    image_polys = img.copy()
                    for psoi_idx in range(len(psoi)):
                        color = np.random.randint(255, size=(1,3))
                        image_polys = psoi[psoi_idx].draw_on_image(bbsoi[psoi_idx].draw_on_image(image_polys, size=5, color=color), alpha_face=0.5, size_points=3, color=color)
                    images_polys.append(ia.imresize_single_image(image_polys, 0.5))
                else:
                    if semantic_segmentation:
                        image_polys = img.copy()
                        for psoi_idx in range(len(psoi)):
                            image_polys = psoi[psoi_idx].draw_on_image(image_polys, alpha_face=0.7, size_points=0, color=color_dict[polygon_data[file_name]['classes'][psoi_idx]], color_lines=color_dict[polygon_data[file_name]['classes'][psoi_idx]])
                        images_polys.append(ia.imresize_single_image(image_polys, 0.5))
                    else:
                        image_polys = img.copy()
                        for psoi_idx in range(len(psoi)):
                            image_polys = psoi[psoi_idx].draw_on_image(bbsoi[psoi_idx].draw_on_image(image_polys, size=5, color=color_dict[polygon_data[file_name]['classes'][psoi_idx]]), alpha_face=0.5, size_points=3, color=color_dict[polygon_data[file_name]['classes'][psoi_idx]])
                        images_polys.append(ia.imresize_single_image(image_polys, 0.5))
        ia.imshow(ia.draw_grid(images_polys, cols=cols))

class VisualizerBoundingBoxes:

    __dataset = None

    def __init__(self, dataset):
        self.__dataset = dataset

    def plot_dataset(self,
        bboxes_data,
        cols=2,
        image_start=0,
        num_of_sample=4,
        color_dict=None,
        randomness=True):

        if randomness:
            samples = random.sample(self.__dataset, num_of_sample)
        else:
            samples = self.__dataset[image_start:num_of_sample]

        images_bboxes = []
        for sample_idx in range(len(samples)):
            for file_idx, file_path in enumerate(glob.glob(samples[sample_idx]['file_name'])):
                file_name = os.path.basename(file_path)
                
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                bbsoi = ia.BoundingBoxesOnImage(bboxes_data[file_name]['bboxes'],
                            shape=img.shape)
                
                if color_dict == None:
                    image_bboxes = img.copy()
                    for bbsoi_idx in range(len(bbsoi)):
                        color = np.random.randint(255, size=(1,3))
                        image_bboxes = bbsoi[bbsoi_idx].draw_on_image(bbsoi[bbsoi_idx].draw_on_image(image_bboxes, size=5, color=color))
                    images_bboxes.append(ia.imresize_single_image(image_bboxes, 0.5))
                else:
                    image_bboxes = img.copy()
                    for bbsoi_idx in range(len(bbsoi)):
                        image_bboxes = bbsoi[bbsoi_idx].draw_on_image(image_bboxes, size=5, color=color_dict[bboxes_data[file_name]['classes'][bbsoi_idx]])
                    images_bboxes.append(ia.imresize_single_image(image_bboxes, 0.5))
        ia.imshow(ia.draw_grid(images_bboxes, cols=cols))