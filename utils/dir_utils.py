from shutil import rmtree, copytree
import os
import numpy as np
from os import listdir
from os.path import isfile, join

def create_target_dirs(dataset_path):
    if os.path.exists(dataset_path):
        rmtree(dataset_path)
    os.makedirs(dataset_path)
    os.mkdir(dataset_path+"/train/")

    labels_dir= dataset_path+"/labels/"
    if not os.path.exists(labels_dir):
            os.mkdir(labels_dir)
    sem_dir= dataset_path+"/semantic_masks/"
    if not os.path.exists(sem_dir):
            os.mkdir(sem_dir)
    return labels_dir, sem_dir

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
    np.save(os.path.join(dataset_path+"/train/", 'indices.npy'), train_inds)
    if(len(val_inds)>0):
        np.save(os.path.join(dataset_path+"/val/", 'indices.npy'), val_inds)

def load_files(dir, first_load = True):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    if(first_load):
        print("number of samples in %s: %d"%(dir.split('/')[-2], len(files)/2))
    return files

def copy_real_data(destination):
    copytree('val/', destination+"/val/")