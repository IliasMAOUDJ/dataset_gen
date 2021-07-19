# Generate depth images from synthetic 3D scenes
This repo is part of my Master 2 internship in LaTIM. The main objective is to generate synthetic depth images for an Object Detection task. 
The code provided uses CAD objects (.stl extension). It uses background images similar to ground truth data and synthetic object with a technique inspired by 
https://arxiv.org/abs/1708.01642

The dataset generated is used for a training on Mask_RCNN (https://github.com/matterport/Mask_RCNN). It is still under Work In Progress.

#TODO:
    - Add requirements.txt

## Installation

```console
# clone the repo
$ git clone https://github.com/IliasMAOUDJ/dataset_gen.git

# change the working directory to dataset_gen
$ cd dataset_gen

####  TODO: add requirements
# install the requirements
$ pip install -r requirements.txt
####
```

## Usage

```console
# for main application
$ python dataset_gen.py
optional arguments:
    --config		config file where you can customize the options of generation

/!\ CAD objects are not available here, you must use yours and save them under ../objects
```





| *Real data* | *Synthetic data* |
|:--:|:--:| 
| ![](https://github.com/IliasMAOUDJ/dataset_gen/blob/main/images/GT.png) | ![](https://github.com/IliasMAOUDJ/dataset_gen/blob/main/images/synthetic.png) |
