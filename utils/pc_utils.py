import numpy as np
import trimesh
import os

def gen_pointcloud(points, min_dist, max_dist, scene_number, nb_points=None):
    pc= trimesh.PointCloud(points[(np.abs(points[:,2])>min_dist) & (np.abs(points[:,2])<max_dist)]) #& (np.abs(points_subsampled[:,2])<2.5)])
    if(nb_points!=None):
        pc = random_sampling(pc, nb_points)
    pc.export('depth/%06d.ply'%scene_number)

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]