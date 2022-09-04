import random
import os
from matplotlib import pyplot as plt

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
        polygons_data,
        opacity=0.5,
        cols=2,
        image_start=0,
        num_of_samples=4,
        color_dict=None,
        without_bboxes=True,
        randomness=True):
        
        if randomness:
            samples = random.sample(self.__dataset, num_of_samples)
        else:
            if (image_start + num_of_samples) > len(self.__dataset):
                print('[!] Warning: num_of_samples argument exceeds the dataset array boundary from image_start argument')
                samples = self.__dataset[image_start:]
            elif image_start > len(self.__dataset):
                print('[!] Warning: image_start argument is greater than the length of the dataset')
                samples = self.__dataset[-1]
            else:
                samples = self.__dataset[image_start:(image_start + num_of_sample)]

        images_polys = []
        for sample in samples:
            file_path = sample['file_name']
            file_name = os.path.basename(file_path)
            
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            psoi = ia.PolygonsOnImage(polygons_data[file_name]['polygons'],
                        shape=img.shape)

            bbsoi = BoundingBoxesOnImage(
                [polygon.to_bounding_box() for polygon in psoi.polygons],
                shape=psoi.shape
            )
            
            if color_dict == None:
                image_polys = img.copy()
                for psoi_idx in range(len(psoi)):
                    color = np.random.randint(255, size=(1,3))
                    if without_bboxes:
                        image_polys = psoi[psoi_idx].draw_on_image(image_polys, alpha_face=opacity, size_points=3, color=color)
                    else:
                        image_polys = psoi[psoi_idx].draw_on_image(bbsoi[psoi_idx].draw_on_image(image_polys, size=5, color=color), alpha_face=opacity, size_points=3, color=color)
                images_polys.append(ia.imresize_single_image(image_polys, 0.5))
            else:
                image_polys = img.copy()
                for psoi_idx in range(len(psoi)):
                    if without_bboxes:
                        image_polys = psoi[psoi_idx].draw_on_image(image_polys, alpha_face=opacity, size_points=0, color=color_dict[polygons_data[file_name]['classes'][psoi_idx]], color_lines=color_dict[polygons_data[file_name]['classes'][psoi_idx]])
                    else:
                        image_polys = psoi[psoi_idx].draw_on_image(bbsoi[psoi_idx].draw_on_image(image_polys, size=5, color=color_dict[polygons_data[file_name]['classes'][psoi_idx]]), alpha_face=opacity, size_points=3, color=color_dict[polygons_data[file_name]['classes'][psoi_idx]])
                images_polys.append(ia.imresize_single_image(image_polys, 0.5))

        ia.imshow(ia.draw_grid(images_polys, cols=cols))

    def visualize_keypoints_of_image(self, image, polygons, color, diameter):
        image = image.copy()

        for keypoints in polygons:
            for (x, y) in keypoints:
                cv2.circle(image, (int(x), int(y)), diameter, color, -1)
        
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(image)
        plt.show()

class VisualizerBoundingBoxes:

    __dataset = None

    def __init__(self, dataset):
        self.__dataset = dataset

    def plot_dataset(self,
        bboxes_data,
        thickness,
        cols,
        image_start,
        num_of_samples,
        color_dict,
        randomness):

        if randomness:
            samples = random.sample(self.__dataset, num_of_samples)
        else:
            if (image_start + num_of_samples) > len(self.__dataset):
                print('[!] Warning: num_of_samples argument exceeds the dataset array boundary from image_start argument')
                samples = self.__dataset[image_start:]
            elif image_start > len(self.__dataset):
                print('[!] Warning: image_start argument is greater than the length of the dataset')
                samples = self.__dataset[-1]
            else:
                samples = self.__dataset[image_start:(image_start + num_of_samples)]

        images_bboxes = []
        for sample in samples:
            file_path = sample['file_name']
            file_name = os.path.basename(file_path)
            
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bbsoi = ia.BoundingBoxesOnImage(bboxes_data[file_name]['bboxes'],
                        shape=img.shape)
            
            if color_dict == None:
                image_bboxes = img.copy()
                for bbsoi_idx in range(len(bbsoi)):
                    color = np.random.randint(255, size=(1,3))
                    image_bboxes = bbsoi[bbsoi_idx].draw_on_image(bbsoi[bbsoi_idx].draw_on_image(image_bboxes, size=thickness, color=color))
                images_bboxes.append(ia.imresize_single_image(image_bboxes, 0.5))
            else:
                image_bboxes = img.copy()
                for bbsoi_idx in range(len(bbsoi)):
                    image_bboxes = bbsoi[bbsoi_idx].draw_on_image(image_bboxes, size=thickness, color=color_dict[bboxes_data[file_name]['classes'][bbsoi_idx]])
                images_bboxes.append(ia.imresize_single_image(image_bboxes, 0.5))
        ia.imshow(ia.draw_grid(images_bboxes, cols=cols))