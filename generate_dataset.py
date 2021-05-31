from __future__ import division
from operator import ge
from matplotlib.pyplot import axis
from numpy.core.defchararray import count
from numpy.lib.twodim_base import tri
import trimesh
import os
from os import listdir
from os.path import isfile, join
import random
from scipy.spatial.transform import Rotation
import numpy as np
import time
import PIL.Image
from trimesh.base import Trimesh
from trimesh.permutate import transform

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def gen_semantic_data(path, scene, scene_number):

    origins, vectors, pixels = scene.camera_rays()

    copy_node = scene.geometry.copy()
    s=1
    arr_geom = []
    px_cnt = np.zeros(2)
    semantic = np.zeros(scene.camera.resolution, dtype=np.uint8)
    depth_map = np.full(scene.camera.resolution,255, dtype=np.uint8)
    for geometry in reversed(copy_node):
        #only bones or boxes
        if(geometry=="wall" or geometry=="floor"):
            continue
        copy_scene = scene.copy()
        for other in copy_node:
            if (other != geometry):
                copy_scene.delete_geometry(other)
        dump = copy_scene.dump(concatenate=True)
        pye = trimesh.ray.ray_pyembree.RayMeshIntersector(dump)
        points, index_ray, _ = pye.intersects_location(
            origins, vectors, multiple_hits=False)
        
        # for each hit, find the distance along its vector
        depth = trimesh.util.diagonal_dot(points - origins[0],
                                        vectors[index_ray])

        if not np.any(depth):
            print("depth is empty")
        # find pixel locations of actual hits
        pixel_ray = pixels[index_ray]
        # convert depth into 0 - 255 uint8
        depth_int = (depth * 255).round().astype(np.uint8)
        e=0
        cnt = 0
        for (i,j) in pixel_ray:              #another solution to try is compute full depth map and then compare if the value is below threshold with numpy (takes less lines)
            if(depth_int[e] < depth_map[i,j]): 
                if("box" in geometry or "tissue" in geometry):
                    semantic[i,j] = 0           
                else:  
                    semantic[i,j] = s
                    cnt +=1              
                depth_map[i,j] = depth_int[e]               
            e+=1
        
        if("box" not in geometry and "tissue" not in geometry):
            px_cnt[s-1]=len(pixel_ray)
            arr_geom.append(geometry)
            s+=1
    
    # create a PIL image from the depth queries
    semantic_map = PIL.Image.fromarray(np.transpose(semantic))
    pixel_values = list(semantic_map.getdata())
    _, counts = np.unique(pixel_values, return_counts=True)
    f = counts[1]/px_cnt[0]
    t = counts[2]/px_cnt[1]
    arr = np.zeros(s-1, dtype =np.bool)
    for j in np.arange(1,arr.size+1):
        if j in pixel_values:
            arr[j-1]=True
    for k in np.arange(arr.size):
        if(arr[k]==False):
            scene.delete_geometry(arr_geom[k])
            print("an object was removed because completely hidden or not enough pixels")
    semantic_map = semantic_map.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    file_name = path+"%06d.png"%(scene_number)
    semantic_map.save(file_name)
    with open(path+"%06d.txt"%scene_number, 'a') as label_scene:
        print("%f"%(f), file=label_scene) #TIBIA
        print("%f"%(t), file=label_scene) #FEMUR

    return f,t

def gen_depth_image(path_pc, path_img, scene, scene_number):

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    dump = scene.dump(concatenate=True)
    # do the actual ray- mesh queries
    pye = trimesh.ray.ray_pyembree.RayMeshIntersector(dump)
    points, index_ray, _ = pye.intersects_location(
        origins, vectors, multiple_hits=False)
    pc= trimesh.PointCloud(points[np.where(np.abs(points[:,2])>0.5)])
    pc.export(path_pc+'/%06d.ply'%i)

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[0],
                                      vectors[index_ray])
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.full(scene.camera.resolution, 255, dtype=np.uint8)
    depth_float = ((depth - depth.min()) / depth.ptp())

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int

    # create a PIL image from the depth queries
    depth_map = PIL.Image.fromarray(np.transpose(a))
    depth_map = depth_map.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    file_name =path_img+"/%06d.png"%(scene_number)
    depth_map.save(file_name)

    return a

def write_data_idx(TRAIN_VAL_SPLIT, nb_samples):
    train_inds=[]
    val_inds=[]
    inds=[]
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

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def gen_scene(scene, num_scene, dir_objects, list_objects):
    dir_path= dataset_path+"labels/"
    if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    labels=[]

    obj = random.choice(list_objects)
    num = obj.split('.')[0].split('_')[1]
    femur = trimesh.load(dir_objects+'Femur_%s.stl'%num)
    tibia = trimesh.load(dir_objects+'Tibia_%s.stl'%num)

    ####################### store half-extents
    fl= (np.amax(femur.vertices[:,0])-np.amin(femur.vertices[:,0])) /2
    fw= (np.amax(femur.vertices[:,1])-np.amin(femur.vertices[:,1])) /2
    fh= (np.amax(femur.vertices[:,2])-np.amin(femur.vertices[:,2])) /2
    tl= (np.amax(tibia.vertices[:,0])-np.amin(tibia.vertices[:,0])) /2
    tw= (np.amax(tibia.vertices[:,1])-np.amin(tibia.vertices[:,1])) /2
    th= (np.amax(tibia.vertices[:,2])-np.amin(tibia.vertices[:,2])) /2
    ####################### get GT centroids pre transform
    fcx = np.amax(femur.vertices[:,0]) + np.amin(femur.vertices[:,0])
    fcy = np.amax(femur.vertices[:,1]) + np.amin(femur.vertices[:,1])
    fcz = np.amax(femur.vertices[:,2]) + np.amin(femur.vertices[:,2])
    tcx = np.amax(tibia.vertices[:,0]) + np.amin(tibia.vertices[:,0])
    tcy = np.amax(tibia.vertices[:,1]) + np.amin(tibia.vertices[:,1])
    tcz = np.amax(tibia.vertices[:,2]) + np.amin(tibia.vertices[:,2])
    
    err_f = np.array([fcx/2,fcy/2,fcz/2])-femur.centroid
    err_t = np.array([tcx/2,tcy/2,tcz/2])-tibia.centroid

    ##################### flexion
    flexion = -np.random.random()*np.pi/2 
    R_femur = rotx(flexion)
    M_flexion= np.eye(4)       # Translation and Rotation
    M_flexion[:3,:3]= R_femur
    M_flexion[3,:]= [0,0,0,1]  
    femur.apply_transform(M_flexion)
    #cyl2.apply_transform(M_flexion)

    ###################### global transform
    angle_x = (np.random.random()*np.pi/5) - np.pi/10 # -18 ~ +18 degree
    angle_y = (np.random.random()*np.pi/3) - np.pi/6
    angle_z = (np.random.random()*2*np.pi) - np.pi
    Rx = rotx(angle_x)
    Ry = roty(angle_y)
    Rz = rotz(angle_z)
    R = np.dot(Rz, np.dot(Ry,Rx))   
    t=np.array([(fcx+tcx)/2, (fcy+tcy)/2, (fcz+tcz)/2])
    t[2]+=(np.random.random()-0.5)*0.3+0.5   #mean dist is 50cm +/- 15
    M_rot_trans= np.eye(4)       # Translation and Rotation
    M_rot_trans[:3,:3]= R
    M_rot_trans[:3,3]=-t
    M_rot_trans[3,:]= [0,0,0,1]
    femur.apply_transform(M_rot_trans)
    tibia.apply_transform(M_rot_trans)
    ############################## get GT centroids
    fcx, fcy, fcz = femur.centroid+np.dot(R, np.dot(R_femur,err_f))
    tcx, tcy, tcz = tibia.centroid+np.dot(R, err_t)

    ##############################
    #"""
    fem_cuisse= trimesh.creation.cylinder(radius=1.2*fl, height=1.2*fw,transform=[[1,0,0,0],[0,1,0,0],[0,0,1,0.04],[0,0,0,1]])
    M_cuisse= np.eye(4)       # Translation and Rotation
    M_cuisse[:3,:3]= np.dot(R,np.dot(R_femur,rotx(-90) ))
    M_cuisse[:3,3]= np.array([fcx,fcy,fcz])
    M_cuisse[3,:]= [0,0,0,1]
    fem_cuisse.apply_transform(M_cuisse)
    femur = femur.difference(fem_cuisse, engine="blender")
    
    #
    tib_mollet= trimesh.creation.cylinder(radius=tl*1.2, height=1.2*tw, transform=[[1,0,0,0],[0,1,0,0],[0,0,1,0.02],[0,0,0,1]])
    M_mollet= np.eye(4)       # Translation and Rotation
    M_mollet[:3,:3]= np.dot(R,rotx(90))#np.dot(R,np.dot(R_femur,rotx(90) ))
    M_mollet[:3,3]= np.array([tcx,tcy,tcz])
    M_mollet[3,:]= [0,0,0,1]
    tib_mollet.apply_transform(M_mollet)
    tibia = tibia.difference(tib_mollet, engine="blender")

    ##############################           add the object to the scene
    scene.add_geometry(femur, geom_name="Femur_%s"%num)
    scene.add_geometry(tibia, geom_name="Tibia_%s"%num)

    """
    other = trimesh.Scene()
    other_obj = trimesh.Scene()
    other_obj.add_geometry(femur)
    other_obj.add_geometry(tibia)
    other.add_geometry(convert_oriented_box_to_trimesh_fmt(2*fl, 2*(0.4*fw+0.02), 2*fh, angle_x+flexion, angle_y, angle_z, femur.centroid))
    other.add_geometry(convert_oriented_box_to_trimesh_fmt(2*tl, 2*(0.2*tw+0.02), 2*th, angle_x, angle_y, angle_z, tibia.centroid))
    mesh_list = trimesh.util.concatenate(other.dump())
    mesh_list.export("bboxes_%d.ply"%num_scene, file_type='ply')

    mesh_list_obj = trimesh.util.concatenate(other_obj.dump())
    mesh_list_obj.export("obj_%d.ply"%num_scene, file_type='ply')
    """
    scene.add_geometry(fem_cuisse, geom_name="tissue_")
    scene.add_geometry(tib_mollet, geom_name="tissue_")
    fcx, fcy, fcz = femur.centroid
    tcx, tcy, tcz = tibia.centroid
    # Write annotation in a txt file for votenet  
    with open(dir_path+"%06d.txt"%num_scene, 'a') as label_scene:
        label_scene.write("Femur %f %f %f %f %f %f %f %f %f\n"%(
                            fcx, fcy, fcz,                                                  #centroid               %f %f %f
                            fl,2*(0.4*fw+0.02),fh,                                                          #length, width, height  %f %f %f
                            angle_x+flexion, angle_y, angle_z,                              #euler angles           %f %f %f
                            ))
        label_scene.write("Tibia %f %f %f %f %f %f %f %f %f\n"%(
                            tcx, tcy, tcz,                           #centroid               %s (%f %f %f)
                            tl,2*(0.4*tw+0.02),th,                                                           #length, width, height  %f %f %f
                            angle_x, angle_y, angle_z,                                  #euler angles           %f %f %f
                            ))
    label_scene.close()
    # Write annotation in a numpy file for Mask RCNN
    labels.append(1)
    labels.append(2)
    np.save(dir_path+"%06d.npy"%(num_scene), labels)
    return flexion, np.array([tcx, tcy, tcz]), np.array([fcx, fcy, fcz])

def heading2rotmat(Rx, Ry, Rz):
        pass
        rotmat = np.zeros((3,3))
        X = rotx(Rx)
        Y = roty(Ry)
        Z = rotz(Rz)
        rotmat = np.dot(Z, np.dot(Y,X))
        return rotmat

def convert_oriented_box_to_trimesh_fmt(l, w, h,Rx, Ry, Rz,ctr):
    lengths = l,w,h
    trns = np.eye(4)
    trns[0:3, 3] = ctr
    trns[3,3] = 1.0            
    trns[:3,:3] = heading2rotmat(Rx, Ry, Rz)
    box_trimesh_fmt = trimesh.creation.box(lengths, trns)
    return box_trimesh_fmt

def gen_additional_objects(scene, pos_tibia, pos_femur):
    p = np.zeros_like(pos_tibia)
    if(np.random.random() >0.5):
        choice = pos_tibia
    else:
        choice = pos_femur
    p[0] = choice[0] + (np.random.random()-0.5)*0.001 
    p[1] = choice[1] + (np.random.random()-0.5)*0.001
    p[2] = -0.2        
    M= np.eye(4)       # Translation and Rotation
    M[:3,3]=p
    M[3,:]= [0,0,0,1]
    geom = trimesh.creation.cylinder(radius=random.random()*0.005+0.005, height= 0.00001, transform=M)
    scene.add_geometry(geom, geom_name="box_")
    #scene.show()

def gen_soup(scene, nb_soup):
   
    for i in range(nb_soup):
        p = (np.random.random(3)-0.5)*0.1
        p[0] = p[0]+ np.sign(p[0])*0.1
        p[2]= -np.random.random()-0.2
        M= np.eye(4)       # Translation and Rotation
        M[:3,:3]= np.eye(3)*0.15
        M[:3,3]=p
        M[3,:]= [0,0,0,1]
        geom = trimesh.creation.random_soup(3)
        geom.apply_transform(M)
        scene.add_geometry(geom, geom_name="box_")   


from shutil import rmtree
def create_target_dirs(dataset_path):

    depth_ims_dir= dataset_path+"depth_ims/"
    if not os.path.exists(depth_ims_dir):
        os.mkdir(depth_ims_dir)
    depth_dir= dataset_path+"depth/"
    if not os.path.exists(depth_dir):
            os.mkdir(depth_dir)
    labels_dir= dataset_path+"labels/"
    if not os.path.exists(labels_dir):
            os.mkdir(labels_dir)
    sem_dir= dataset_path+"semantic_masks/"
    if not os.path.exists(sem_dir):
            os.mkdir(sem_dir)
    calib_dir = dataset_path+"calib/"
    if not os.path.exists(calib_dir):
            os.mkdir(calib_dir)
    return depth_ims_dir, depth_dir, labels_dir, sem_dir, calib_dir



#TODO: gridsearch, same metrics, base de données bruitée
if __name__ == '__main__':
    tic = time.perf_counter()
    print("---------  Generating dataset scenes ---------")
    nb_samples=600
    print("number of scenes: %d"%nb_samples)
    TRAIN_VAL_SPLIT = 0.90
    print("train: %d \t val: %d"%(nb_samples*TRAIN_VAL_SPLIT, np.round(nb_samples*(1-TRAIN_VAL_SPLIT))))
    stats = {'Femur':0, 'Tibia': 0}

    total_flexion = []
    total_pixels = []
    total_femur_dist = []
    total_tibia_dist = []
    total_occlusion_femur = []
    total_occlusion_tibia = []
    dataset_path = ROOT_DIR+"/test_noise/"
    if os.path.exists(dataset_path):
        rmtree(dataset_path)
    os.makedirs(dataset_path)
    depth_ims_dir, depth_dir, labels_dir, sem_dir, calib_dir= create_target_dirs(dataset_path)
    
    train_dir = ROOT_DIR + "/objects/train/"
    train_objects =  [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
    print("number of train samples: %d"%(len(train_objects)/2))
    val_dir = ROOT_DIR + "/objects/val/"
    val_objects =  [f for f in listdir(val_dir) if isfile(join(val_dir, f))]
    print("number of val samples: %d"%(len(val_objects)/2))

    resolution=[960,600]
    cam_trns= np.eye(4,4)
    cam = trimesh.scene.Camera("cam", resolution=resolution, fov=[39,25])

    for i in range(nb_samples):
        if(i==0):
            print("--------- Generating train data ---------")
        if(i==int(TRAIN_VAL_SPLIT*nb_samples)):
            print("---------- Generating val data ----------")
        print("%06d"%i)
        #cam_trns[:3,3]=[0,0,np.random.random()*0.2+0.4]
        scene = trimesh.Scene(camera=cam, camera_transform=cam_trns)
        with open(calib_dir+"%06d.txt"%i, 'w') as calib_cam:
            cam_int = np.eye(3)
            cam_ext = scene.camera.K
            np.savetxt(calib_cam, (cam_int.reshape(-1), cam_ext.reshape(-1)), fmt='%g')
        calib_cam.close()

        M_wall= np.eye(4,4)
        Ry = roty(np.random.random()*np.pi/3-np.pi/6)
        Rx = rotx(np.random.random()*np.pi/3-np.pi/6)
        M_wall[:3,:3]= np.dot(Ry, Rx)
        M_wall[:3,3]=[0,0,-np.random.random()*0.3-0.9]
        M_wall[3,:]= [0,0,0,1]
        wall = trimesh.creation.box([5,5,0.005], M_wall)
        scene.add_geometry(wall, geom_name="wall")
        
        if(i< int(TRAIN_VAL_SPLIT*nb_samples)):
            flexion, pos_tibia, pos_femur = gen_scene(scene, i, train_dir, train_objects)
        else:
            flexion, pos_tibia, pos_femur = gen_scene(scene, i, val_dir, val_objects)
        total_flexion.append(flexion)
       
        for geometry in scene.geometry:
            if "wall" not in geometry and "box" not in geometry and "tissue" not in geometry:
                if "Femur" in geometry:
                    stats['Femur']+=1
                elif "Tibia" in geometry:
                    stats['Tibia']+=1
                if geometry.split(':')[0] in stats.keys():
                    stats[geometry.split(':')[0]]+=1
                else:
                    stats[geometry.split(':')[0]]=1

        gen_additional_objects(scene, pos_tibia, pos_femur)
        #gen_soup(scene, 4)
        occlusion_f, occlusion_t = gen_semantic_data(sem_dir, scene, i)
        pixels = gen_depth_image(depth_dir, depth_ims_dir, scene,i)   
       
        total_pixels.append(pixels)
        total_femur_dist.append(np.sqrt(pos_femur.dot(pos_femur)))
        total_tibia_dist.append(np.sqrt(pos_tibia.dot(pos_tibia)))
        total_occlusion_femur.append(occlusion_f)
        total_occlusion_tibia.append(occlusion_t)



    mean = np.mean(total_pixels)
    mean_flexion = np.mean(total_flexion)
    mean_dist_f = np.mean(total_femur_dist)
    mean_dist_t = np.mean(total_tibia_dist)
    mean_occ_t = np.mean(total_occlusion_tibia)
    mean_occ_f = np.mean(total_occlusion_femur)
    write_data_idx(TRAIN_VAL_SPLIT,nb_samples)

    with open(dataset_path+'stats.txt', 'w') as f:
        import collections
        stats = collections.OrderedDict(sorted(stats.items()))
        for k, v in stats.items():
            print("%s: %s"%(k,v), file=f)
        print("\nmean_pixel: %f"%(mean), file=f)
        print("mean_flexion: %f"%(flexion), file=f)
        print("(distance from centroid)", file=f)
        print("mean_distance_tibia: %f"%(mean_dist_t), file=f)
        print("mean_distance_femur: %f"%(mean_dist_f), file=f)
        print("occlusion_femur: %f"%(mean_occ_f), file=f )
        print("occlusion_tibia: %f"%(mean_occ_t), file=f)
    toc = time.perf_counter()
    print("Generating dataset took %f seconds"%(toc-tic))