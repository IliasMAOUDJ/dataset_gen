from __future__ import division
import trimesh
import os
import numpy as np
from utils.scene_utils import *
from utils.stats_utils import *
from utils.img_utils import *
from utils.dir_utils import *
from utils.pc_utils import *
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+ "/.."

def gen_camera_param(resolution, fov):
    cam_trns= np.eye(4,4)
    cam = trimesh.scene.Camera("cam", resolution=resolution, fov=fov, z_near=0.05, z_far=500)
    return cam, cam_trns


def generate(config):
    NB_SAMPLES = config['dataset']['nb_samples']
    TRAIN_VAL_SPLIT = config['dataset']['train_val_split']
    DATASET_NAME = config['dataset']['name']
    TRAIN_DIR = config['dataset']['train_dir']
    VAL_DIR = config['dataset']['val_dir']

    resolution = config['camera']['resolution']
    fov = config['camera']['fov']
    gen_pc = config['point_cloud']['generate']
    if(gen_pc):
        nb_points = config['point_cloud']['nb_points']
        min_dist = config['point_cloud']['min_distance']
        max_dist = config['point_cloud']['max_distance']

    img_output_dim = config['image']['resolution']
    bg_dir = config['image']['background_path']
    far = config['scene']['far']
    near = config['scene']['near'] 
    dist_objects = config['scene']['objects_distance']
    objects=config['objects']
    print("---------  Generating dataset scenes ---------")
    print("number of scenes: %d"%NB_SAMPLES)
    print("train: %d \t val: %d"%(NB_SAMPLES*TRAIN_VAL_SPLIT, np.round(NB_SAMPLES*(1-TRAIN_VAL_SPLIT))))  
    stats = {}; total_flexion = [] ;  total_pixels = []; total_occlusion_femur = [] ; total_occlusion_tibia = [] ; total_occlusion_guide = []    

    dataset_path = "../"+DATASET_NAME
    labels_dir, sem_dir= create_target_dirs(dataset_path)

    train_objects = load_files(TRAIN_DIR)
    val_objects =  load_files(VAL_DIR)

    camera, camera_transform = gen_camera_param(resolution=resolution, fov=fov)
    i=0
    pbar =tqdm(total=NB_SAMPLES)
    while i < NB_SAMPLES:        
        scene = init_scene(camera, camera_transform, far)
        if(i< int(TRAIN_VAL_SPLIT*NB_SAMPLES)):
            if(i==0):
                print("--------- Generating train data ---------")
            flexion, positions = gen_scene(scene, TRAIN_DIR, train_objects, objects,dist_objects, v1=True)
            is_training_data = True
        else:
            if(i==int(TRAIN_VAL_SPLIT*NB_SAMPLES)):
                print("---------- Generating val data ----------")
            flexion, positions = gen_scene(scene, VAL_DIR, val_objects, objects, dist_objects, v1=True)
            is_training_data = False
  
        #generate the closest object in the scene to get a proper depth representation
        gen_additional_objects(scene, position=positions, near=near)
        pixels, points = gen_depth_image(dataset_path, scene,i, img_output_dim, bg_dir, train=is_training_data)   
        if(gen_pc):
            gen_pointcloud(points, min_dist, max_dist, i, nb_points)

        occlusion_f, occlusion_t, occlusion_g = gen_semantic_data(sem_dir, scene, i, labels_dir, img_output_dim)

        append_values(total_flexion, flexion); append_values(total_pixels, pixels); 
        append_values(total_occlusion_femur, occlusion_f); append_values(total_occlusion_guide, occlusion_g); append_values(total_occlusion_tibia, occlusion_t)
        for geometry in scene.geometry:
            if "Femur" in geometry or "Tibia" in geometry or "Guide" in geometry:
                try:
                    stats[geometry.split(':')[0]]+=1
                except:
                    stats[geometry.split(':')[0]]=1
        pbar.update()
        i+=1
    pbar.close()
    compute_mean_values(stats, total_pixels, total_flexion, total_occlusion_tibia, total_occlusion_femur, total_occlusion_guide)
    write_data_idx(TRAIN_VAL_SPLIT,NB_SAMPLES, dataset_path)
    write_stats(dataset_path, stats)

    copy_real_data(dataset_path)

import argparse
from autolab_core import YamlConfig
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='param.yaml')
    config = YamlConfig('param.yaml')
    config.save(os.path.join('./', config['save_conf_name']))
    generate(config)
    