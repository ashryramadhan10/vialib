# Vialib

Vialib is an open source library written in Python for helping you to do image augmentation, visualize your dataset annotations (in polygon or bounding box), convert to various most used formats in computer vision model such as YOLOv4, YOLOv5, Pascal VOC, ICCV09, COCO 2017, and Semantic Segmentation Mask from Vgg Image Annotator (VIA) version 2.0.11 JSON files format.

**Supported annotation styles**:
* Bounding Box
* Polygon

**Convert to format**:
* YOLO - YOLOv4 Format
* YOLOv5 - YOLOv7 Format
* Pascal VOC Format
* ICCV2009 Format (Standford Background Dataset)
* COCO 2017 Format
* Semantic Segmentation Mask (Binary Class or Multi Class)

**Image augmentation engines**:
* imgaug
* albumentations

**Visualization**:
* Polygons (support multiclass)
* Bounding Boxes (support multiclass)
* Keypoints (support multiclass)

# Tutorial

## 1. First you need to specify your dataset same with this folder structure, e.g:
### Folder Structure
```
C:\DATASET
    image_1.jpg
    image_2.jpg
    image_3.jpg
    ...
    annotation.json
```

## 2. Read your dataset based on what kind of shape or format did you used before (polygons or bounding boxes), e.g in polygons:

### Polygons Format
```python
from vialib.dataset import DatasetPolygon

dataset = DatasetPolygon(input_dir='DATASET/', output_dir='OUTPUT/', annotation_file_name='annotation.json')
```

### Bounding Boxes Format
```python
from vialib.dataset import DatasetBoundingBox

dataset = DatasetBoundingBox(input_dir='DATASET/', output_dir='OUTPUT/', annotation_file_name='annotation.json')
```

## 3. Getting information from your dataset by checking the length of your dataset or your annotaiton data, here some function to get information from your dataset:

### Polygons Format
```python
print(dataset.getVIAJSON())
print(dataset.getDataset())
print(dataset.getDatasetDict())
print(dataset.getPolygonData())
print(dataset.printClasses())
```

### Bounding Boxes Format
```python
print(dataset.getVIAJSON())
print(dataset.getDataset())
print(dataset.getDatasetDict())
print(dataset.getBoundingBoxes())
print(dataset.printClasses())
```
## 4. Visualize your dataset, you can use `plot_dataset`, this function has arguments:
### Polygons Format
* `opacity` -> **float** (0 - 1): opactiy of the mask
* `cols` -> **int** : number of columns
* `image_start` -> **int** : (you can use this when randomenss is False), starting index to plot image
* `num_of_sample` -> **int** : number of images you want to display
* `color_dict` -> **dict** : color dictionary if you want to display your dataset with specific color based on its class
* `without_bboxes` -> **boolean** : show without bounding boxes
* `randomness` -> **int** : chose image randomly to be displayed

example:
```python
# display randomly, no color dictionary, and no bounding bboxes
dataset.plot_dataset(opacity=0.5, cols=10, num_of_samples=10, without_bboxes=True)

# display with color dictionary
color_dict = {
    'class_1': [255, 0, 0],
    'class_2': [0, 255, 0],
    'class_3': [0, 0, 255],
}
dataset.plot_dataset(opacity=0.5, cols=10, num_of_samples=10, color_dict=color_dict, without_bboxes=True)

# display with bounding boxes
dataset.plot_dataset(opacity=0.5, cols=10, num_of_samples=10, color_dict=color_dict, without_bboxes=False)

# display in consecuitve manner (randomness = False)
dataset.plot_dataset(opacity=0.5, cols=5, image_start=0, num_of_samples=10, color_dict=color_dict, without_bboxes=True, randomness=False)
```
### Bounding Boxes Format
same with polygons but there is no `opacity`, this argument is changed with `thickness`
* `thickness` -> **int** : the thickness of bounding box
* `cols` -> **int** : number of columns
* `image_start` -> **int** : (you can use this when randomenss is False), starting index to plot image
* `num_of_sample` -> **int** : number of images you want to display
* `color_dict` -> **dict** : color dictionary if you want to display your dataset with specific color based on its class
* `without_bboxes` -> **boolean** : show without bounding boxes
* `randomness` -> **int** : chose image randomly to be displayed
example:
```python
# display randomly, no color dictionary, and no bounding bboxes
dataset.plot_dataset(thickness=5, cols=10, num_of_samples=10)

# display with color dictionary
color_dict = {
    'class_1': [255, 0, 0],
    'class_2': [0, 255, 0],
    'class_3': [0, 0, 255],
}
dataset.plot_dataset(thickness=5, cols=10, num_of_samples=10, color_dict=color_dict)

# display in consecuitve manner (randomness = False)
dataset.plot_dataset(thickness=5, cols=5, image_start=0, num_of_samples=10, color_dict=color_dict, randomness=False)
```
## 5. Convert to various formats
this package can convert VIA json file to various format, e.g:
```python
# convert to YOLOv4 format
dataset.convert_to_yolo_format()

# convert to YOLOv5 format
class_map = {
    0: 'class_1',
    1: 'class_2',
    3: 'class_3',
    ...
    n: 'class_n',
}
dataset.convert_to_yolov5_format(class_map)

# convert to Pascal VOC format
dataset.convert_to_pascalvoc_format()

# convert to Binary Semantic Segmentation Mask format
dataset.convert_to_binary_semantic_segmentation_format()

# convert to ICCV09 format (Multiclass Semantic Segmentation Mask)
class_color_dict = {
    'class_1': {'color': [255, 0, 0], 'index': 0},
    'class_2': {'color': [0, 255, 0], 'index': 1},
    'class_3': {'color': [0, 0, 255], 'index': 2},
    ...
    'class_n': {'color': [(0-255), (0-255), (0-255)], 'index': (uint)},
}
dataset.convert_to_iccv09_format(class_color_dict)
```
## 6. Augmentation and Transforming Dataset
**Augmentation**: augmenting your dataset using [albumentations](https://albumentations.ai/) or [imgaug](https://imgaug.readthedocs.io/en/latest/), e.g:
### Augmentation
```python
import albumentations as A
import imgaug.augmenters as iaa

augment_albumentations = A.Compose([
    ... # here your transformation functions
])
dataset.augment(aug=augment_albumentations, aug_engine='albumentations', repeat=1)

augment_imgaug = iaa.Sequential([
... # here your transformation functions
])
dataset.augment(aug=augment_imgaug, aug_engine='imgaug', repeat=1)
```
> `augment` function arguments:
> * `aug` -> A.Compose or iaa.Sequential object
> * `aug_engine` -> str : 'albumentations' or 'imgaug augmentation package
> * `repeat` -> unsigned int : to generate more images based on how many times this argument is set

### Transform
```python
import albumentations as A
import imgaug.augmenters as iaa

transform_albumentations = A.Compose([
    ... # here your transformation functions
])
dataset.transform(tf=transform_albumentations, aug_engine='albumentations', add_name='', numeric_file_name=False, repeat=1)

transform_imgaug = iaa.Sequential([
    ... # here your transformation functions
])
dataset.transform(tf=transform_imgaug, aug_engine='imgaug', add_name='', numeric_file_name=False, repeat=1)
```
> `transform` function arguments:
> * `tf` -> A.Compose or iaa.Sequential object
> * `aug_engine` -> str : 'albumentations' or 'imgaug augmentation package
> * `add_name` -> str: (optionaly, default is '') if you want to add prefix to your images filename
> * `numeric_file_name` -> boolean: if you set this True, your images filename will change to numerical format (1 to n where n is nb. of your dataset)
> * `repeat` -> unsigned int : to generate more images based on how many times this argument is set

## 7. Split your Dataset into Train and Test
`split_train_test` function arguments:
* `class_list` -> list : list of your object classes
* `regex_list` -> list : list of your regex list to get your object classes
* `train_ratio` -> float (0 - 1) : your train dataset ratio
```python
class_list = ['dataset', 'dataset', 'dataset']
regex_list = [r'class_1_\d+\.jpg', r'class_2_\d+\.jpg', r'class_3_\d+\.jpg']
dataset.split_train_test(class_list=class_list, regex_list=regex_list, train_ratio=0.8)

# or

class_list = ['class_1', 'class_2', 'class_3']
regex_list = [r'class_1_\d+\.jpg', r'class_2_\d+\.jpg', r'class_3_\d+\.jpg']
dataset.split_train_test(class_list=class_list, regex_list=regex_list, train_ratio=0.8)
```

## 8. Merging VIA JSON Files
you can merge your VIA JSON files together using `merge` function, here the function arguments:
* `dataset_list` -> VIA JSON object: 
```python
dataset1 = DatasetPolygon('...')
dataset2 = DatasetPolygon('...')
dataset3 = DatasetPolygon('...')

dataset_list = [dataset2.getVIAJSON(), dataset3.getVIAJSON()]

dataset1.merge(dataset_list=dataset_list)
```
