from shutil import rmtree
import os
import numpy as np
from os import listdir
from os.path import isfile, join

def create_target_dirs(dataset_path):
    if os.path.exists(dataset_path):
        rmtree(dataset_path)
    os.makedirs(dataset_path)
    depth_ims_dir= dataset_path+"/depth_ims/"
    if not os.path.exists(depth_ims_dir):
        os.mkdir(depth_ims_dir)
    depth_dir= dataset_path+"/depth/"
    if not os.path.exists(depth_dir):
            os.mkdir(depth_dir)
    labels_dir= dataset_path+"/labels/"
    if not os.path.exists(labels_dir):
            os.mkdir(labels_dir)
    sem_dir= dataset_path+"/semantic_masks/"
    if not os.path.exists(sem_dir):
            os.mkdir(sem_dir)
    calib_dir = dataset_path+"/calib/"
    if not os.path.exists(calib_dir):
            os.mkdir(calib_dir)
    return depth_ims_dir, depth_dir, labels_dir, sem_dir, calib_dir

def write_data_idx(TRAIN_VAL_SPLIT, nb_samples, dataset_path):
    train_inds=[]
    val_inds=[]
    for count in range(nb_samples):
        # split data in train and val set
        part= count/nb_samples
        if(part<TRAIN_VAL_SPLIT):
            train_inds.append(count)
        else :
            val_inds.append(count)
    np.save(os.path.join(dataset_path, 'train_indices.npy'), train_inds)
    np.save(os.path.join(dataset_path, 'val_indices.npy'), val_inds)
    np.save(os.path.join(dataset_path, 'indices.npy'), train_inds+val_inds)

def load_files(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    print("number of samples in %s: %d"%(dir.split('/')[-2], len(files)/2))
    return files