# Generate depth images from synthetic 3D scenes
Project using Trimesh (https://github.com/mikedh/trimesh)

This repo is part of my Master 2 internship in LaTIM. The main objective is to generate synthetic depth images for an Object Detection task. 
The code provided uses CAD objects (.stl extension). It uses background images similar to ground truth data and synthetic object with a technique inspired by 
"Dwibedi, Debidatta, Ishan Misra and M. Hebert. “Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection.” 2017 IEEE International Conference on Computer Vision (ICCV) (2017): 1310-1319."

THe dataset generated is used for a training on Mask_RCNN (https://github.com/matterport/Mask_RCNN). It is still under Work In Progress.

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
    --number,                                      Number of samples to generate (default: 2000)
    --name,                                        Name of the dataset (default: datetime)
    --camera_resolution                            Camera resolution (default: [1024,1024]) Kinect Azure 2 WFOV
    --camera_fov                                   Camera fov (default: [60,60]) Kinect Azure 2 WFOV
    --image_output_dim                             Output dimension of generated images, this crops the left/top/right/bottom (default: [640,640])

/!\ CAD objects are not available here, you must use yours and save them under ../objects
```





| *Real data* | *Synthetic data* |
|:--:|:--:| 
| ![](https://github.com/IliasMAOUDJ/dataset_gen/blob/main/images/GT.png) | ![](https://github.com/IliasMAOUDJ/dataset_gen/blob/main/images/synthetic.png) |
