from __future__ import division
from skimage.data import camera
import trimesh
import os
import numpy as np
from utils.scene_utils import *
from utils.stats_utils import *
from utils.img_utils import *
from utils.dir_utils import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+ "/.."

def gen_camera_param(resolution, fov):
    cam_trns= np.eye(4,4)
    cam = trimesh.scene.Camera("cam", resolution=resolution, fov=fov, z_near=0.05, z_far=500)
    #cam_trns[:3,3]=[0,0,np.random.random()*0.2+0.4]
    
    #with open(calib_dir+"%06d.txt"%i, 'w') as calib_cam:
    #    cam_int = np.eye(3)
    #    cam_ext = scene.camera.K
    #    np.savetxt(calib_cam, (cam_int.reshape(-1), cam_ext.reshape(-1)), fmt='%g')
    #calib_cam.close()
    return cam, cam_trns


def generate(config):
    nb_samples = config['dataset']['nb_samples']
    TRAIN_VAL_SPLIT = config['dataset']['train_val_split']
    dataset_name = config['dataset']['name']
    train_dir = config['dataset']['train_dir']
    val_dir = config['dataset']['val_dir']

    resolution = config['camera']['resolution']
    fov = config['camera']['fov']

    img_output_dim = config['image']['resolution']
    
    print("---------  Generating dataset scenes ---------")
    print("number of scenes: %d"%nb_samples)
    print("train: %d \t val: %d"%(nb_samples*TRAIN_VAL_SPLIT, np.round(nb_samples*(1-TRAIN_VAL_SPLIT))))
    
    stats = {}; total_flexion = [] ;  total_pixels = []; total_occlusion_femur = [] ; total_occlusion_tibia = []    

    dataset_path = "../"+dataset_name
    depth_ims_dir, depth_dir, labels_dir, sem_dir, _= create_target_dirs(dataset_path)

    train_objects = load_files(train_dir)
    val_objects =  load_files(val_dir)

    camera, camera_transform = gen_camera_param(resolution=resolution, fov=fov)
    for i in tqdm(range(nb_samples)):        
        scene = init_scene(camera, camera_transform)
        if(i< int(TRAIN_VAL_SPLIT*nb_samples)):
            if(i==0):
                print("--------- Generating train data ---------")
            flexion, pos_femur, _ = gen_scene(scene, i, train_dir, train_objects, labels_dir)
        else:
            if(i==int(TRAIN_VAL_SPLIT*nb_samples)):
                print("---------- Generating val data ----------")
            flexion, pos_femur, _ = gen_scene(scene, i, val_dir, val_objects, labels_dir)
        
        for geometry in scene.geometry:
            if "Femur" in geometry or "Tibia" in geometry:
                try:
                    stats[geometry.split(':')[0]]+=1
                except:
                    stats[geometry.split(':')[0]]=1
        gen_additional_objects(scene, position=pos_femur)
        pixels = gen_depth_image(ROOT_DIR, depth_dir, depth_ims_dir, scene,i, img_output_dim)   
        occlusion_f, occlusion_t = gen_semantic_data(sem_dir, scene, i, img_output_dim)
        append_values(total_flexion, flexion); append_values(total_pixels, pixels); append_values(total_occlusion_femur, occlusion_f); append_values(total_occlusion_tibia, occlusion_t)

    compute_mean_values(stats, total_pixels, total_flexion, total_occlusion_tibia, total_occlusion_femur)
    write_data_idx(TRAIN_VAL_SPLIT,nb_samples, dataset_path)
    write_stats(dataset_path, stats)

from tqdm import tqdm
import argparse
from datetime import datetime
from autolab_core import YamlConfig
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='param.yaml')
    config = YamlConfig('param.yaml')
    config.save(os.path.join('./', config['save_conf_name']))
    generate(config)