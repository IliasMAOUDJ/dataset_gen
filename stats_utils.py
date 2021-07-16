import numpy as np
import collections
def append_values(array, value):
    array.append(value)

def compute_mean_values(stats, pix, flex, occ_tibia, occ_femur):
    stats['mean_flexion_angle'] = np.mean(flex)
    stats['mean_pixel'] = np.mean(pix)
    stats['mean_tibia_occlusion'] = np.mean(occ_tibia)
    stats['mean_tibia_occlusion'] = np.mean(occ_femur)


def write_stats(dataset_path, stats):
    with open(dataset_path+'stats.txt', 'w') as f:  
        stats = collections.OrderedDict(sorted(stats.items()))
        for k, v in stats.items():
            print("%s: %s"%(k,v), file=f)