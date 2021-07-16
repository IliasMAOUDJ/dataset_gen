from __future__ import division
import trimesh
import os
from os import listdir, write
from os.path import isfile, join
import random
from scipy.spatial.transform import Rotation
import numpy as np
import PIL.Image
from dir_utils import create_target_dirs, load_files, write_data_idx
from scene_utils import rotx, rotz, roty, normalize, init_scene
from stats_utils import append_values, compute_mean_values, write_stats

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+ "/.."

def gen_semantic_data(path, scene, scene_number, resolution):

    origins, vectors, pixels = scene.camera_rays()

    copy_node = scene.geometry.copy()
    arr_geom = []
    semantic = np.zeros(scene.camera.resolution, dtype=np.uint8)
    depth_map = np.full(scene.camera.resolution,255, dtype=np.uint8)
    px_cnt = {}
    for geometry in reversed(copy_node):
        #only bones or boxes
        if(geometry=="wall" or "clutter" in geometry):
            continue
        if "Tibia" in geometry:
            s=2
        elif "Femur" in geometry:
            s=1
        elif "tool" in geometry:
            s=3
        else:
            s=0
        copy_scene = scene.copy()
        for other in copy_node:
            if (other != geometry):
                copy_scene.delete_geometry(other)
        dump = copy_scene.dump(concatenate=True)
        pye = trimesh.ray.ray_pyembree.RayMeshIntersector(dump)
        try:
            points, index_ray, _ = pye.intersects_location(origins, vectors, multiple_hits=False)
        except:
            print("intersect error")

        
        # for each hit, find the distance along its vector
        depth = trimesh.util.diagonal_dot(points - origins[0],
                                        vectors[index_ray])

        if not np.any(depth):
            continue
        # find pixel locations of actual hits
        pixel_ray = pixels[index_ray]
        # convert depth into 0 - 255 uint8
        depth_int = (depth * 255).round().astype(np.uint8)

        #if object is closer, the semantic map takes its id value, else we keep the original value
        semantic[pixel_ray[:,0], pixel_ray[:,1]] = np.array([s  if b <= a else o for (a, b, o) in zip(depth_map[pixel_ray[:,0],pixel_ray[:,1]], depth_int, semantic[pixel_ray[:,0], pixel_ray[:,1]])])
        #keep the closest object in memory
        depth_map[pixel_ray[:,0], pixel_ray[:,1]] = np.array([min(a, b) for (a, b) in zip(depth_map[pixel_ray[:,0],pixel_ray[:,1]], depth_int)])

        if("Femur" in geometry or "Tibia" in geometry):
            px_cnt[geometry.split('_')[0]]=len(pixel_ray)
            arr_geom.append(geometry)
           
    # create a PIL image from the depth queries
    semantic_map = PIL.Image.fromarray(np.transpose(semantic))
    pixel_values = list(semantic_map.getdata())
    unique, counts = np.unique(pixel_values, return_counts=True)
    t,f = 0, 0
    for(k,v) in zip(unique, counts):
        if(k==1):
            f = v/px_cnt['Femur'] 
        if(k ==2):
            t = v/px_cnt['Tibia']

    semantic_map = semantic_map.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    file_name = path+"%06d.png"%(scene_number)

    background = PIL.Image.fromarray(np.zeros((1024,1024)),mode='L')
    bg_w, bg_h = background.size
    background.paste(semantic_map)
    w, h = resolution

    left = (bg_w-w)//2
    upper = (bg_h-h)//2
    right = bg_w - (bg_w-w)//2
    lower = bg_h - (bg_h-h)//2
    background = background.crop((left,upper,right,lower))

    background.save(file_name)
    with open(path+"%06d.txt"%scene_number, 'a') as label_scene:
        print("%f"%(f), file=label_scene) #TIBIA
        print("%f"%(t), file=label_scene) #FEMUR

    return f,t

from skimage import io
def gen_depth_image(path_pc, path_img, scene, scene_number, resolution):

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    dump = scene.dump(concatenate=True)
    # do the actual ray- mesh queries
    pye = trimesh.ray.ray_pyembree.RayMeshIntersector(dump)
    points, index_ray, _ = pye.intersects_location(
        origins, vectors, multiple_hits=False)

    #pc= trimesh.PointCloud(points[(np.abs(points[:,2])>0.31) & (np.abs(points[:,2])<1.35)]) #& (np.abs(points_subsampled[:,2])<2.5)])
    #pc.export(path_pc+'/%06d.ply'%i)

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[0],
                                      vectors[index_ray])
    #depth[np.random.choice(depth.shape[0], int(0.02*depth.shape[0]))] = 0
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.full(scene.camera.resolution, 0, dtype=np.uint8)
    MAX_DEPTH = 90
    try:
        depth_float = ((depth - depth.min()) / depth.ptp())
    except:   
        print(len(depth))
        print(depth)
        scene.show()
    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * MAX_DEPTH).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    


    background = io.imread(ROOT_DIR+'/background_image/'+random.choice(os.listdir(ROOT_DIR+'/background_image/')))
    background[background>90]=0
    background_img = PIL.Image.fromarray(background)
    

    a= normalize(a)
    a[a>MAX_DEPTH]=0 #np.random.random()*200
    
    # create a PIL image from the depth queries
    depth_map = PIL.Image.fromarray(np.transpose(a))
    depth_map = depth_map.transpose(PIL.Image.FLIP_TOP_BOTTOM) #flip to be in the "camera view"
    #depth_map.rotate(np.random.random()*360)

    bg_w, bg_h = background.shape
    w, h = resolution

    left = (bg_w-w)//2
    upper = (bg_h-h)//2
    right = bg_w - (bg_w-w)//2
    lower = bg_h - (bg_h-h)//2

    mask_array= np.where(np.transpose(a)==0, np.transpose(a), 255)
    mask_img = PIL.Image.fromarray(mask_array).transpose(PIL.Image.FLIP_TOP_BOTTOM)
    background_img = background_img.rotate(np.random.random()*360)
    background_img.paste(depth_map,mask=mask_img)
    background_img = background_img.crop((left,upper,right,lower))
   

    file_name =path_img+"/%06d.png"%(scene_number)
    background_img.save(file_name)
    #depth_map.save(file_name)
    return np.asfarray(background_img)

def gen_scene(scene, num_scene, dir_objects, list_objects, labels_dir):
    labels=[]

    obj = random.choice(list_objects)
    num = obj.split('.')[0].split('_')[1]
    if "13" in num:
        num="027"
    femur = trimesh.load(dir_objects+'Femur_%s.stl'%num)
    tibia = trimesh.load(dir_objects+'Tibia_%s.stl'%num)
    ####################### store half-extents
    fl, fw, fh = femur.extents/2
    tl, tw, th = femur.extents/2
    ####################### get GT centroids pre transform
    fcx, fcy, fcz = np.mean(femur.bounds, axis=0)
    tcx, tcy, tcz = np.mean(tibia.bounds, axis=0)
   
    err_f = np.array([fcx,fcy,fcz])-femur.centroid
    err_t = np.array([tcx,tcy,tcz])-tibia.centroid

    ##################### flexion
    flexion = -np.pi/3-(np.random.random()-0.5)*np.pi/2
    R_femur = rotx(flexion)
    M_flexion= np.eye(4)       # Translation and Rotation
    M_flexion[:3,:3]= R_femur
    M_flexion[3,:]= [0,0,0,1]  
    femur.apply_transform(M_flexion)

    ###################### global transform
    
    angle_x = 0 #(np.random.random()*np.pi/3) - np.pi/6 # -18 ~ +18 degree
    angle_y = (np.random.random()*np.pi) - np.pi/2
    angle_z = (np.random.random()*2*np.pi) - np.pi
    Rx = rotx(angle_x)
    Ry = roty(angle_y)
    Rz = rotz(angle_z)
    R = np.dot(Rz, np.dot(Ry,Rx))  
    t=np.array([(np.random.random()-0.5)*0.2,(np.random.random()-0.5)*0.2,0]) 
    
    #t=np.array([(fcx+tcx)/2, (fcy+tcy)/2, 0])
    t[2]+=(np.random.random()-0.5)*0.3+0.7  #mean dist is 70cm +/- 15
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
    """
    bg= trimesh.creation.cylinder(radius=1.5*fw, height=1.8*fh)
    M_bg= np.eye(4)
    M_bg[:3,3]= np.array([fcx,fcy,fcz-0.05])
    M_bg[3,:]= [0,0,0,1]
    bg.apply_transform(M_bg)
    """

    fem_cuisse= trimesh.creation.cylinder(radius=1.5*fw, height=1.8*fh)
    vec = np.dot(np.dot(R,np.dot(R_femur,rotx(-90))), np.array([0,0,fh]))
    M_cuisse= np.eye(4)       # Translation and Rotation
    M_cuisse[:3,:3]= np.dot(R,np.dot(R_femur,rotx(-90) ))
    M_cuisse[:3,3]= np.array([fcx,fcy,fcz]) + vec
    M_cuisse[3,:]= [0,0,0,1]
    fem_cuisse.apply_transform(M_cuisse)
    femur = femur.difference(fem_cuisse, engine="blender")
    
    #
    """
    tib_mollet= trimesh.creation.cylinder(radius=tw*1.4, height=1.8*th, transform=[[1,0,0,0],[0,1,0,0],[0,0,1,th],[0,0,0,1]])
    M_mollet= np.eye(4)       # Translation and Rotation
    M_mollet[:3,:3]= np.dot(R,rotx(90))
    M_mollet[:3,3]= np.array([tcx,tcy,tcz])
    M_mollet[3,:]= [0,0,0,1]
    tib_mollet.apply_transform(M_mollet)
    tibia = tibia.difference(tib_mollet, engine="blender")
    """
    ##############################           add the object to the scene
    scene.add_geometry(femur, geom_name="Femur_%s"%num)
    #scene.add_geometry(tibia, geom_name="Tibia_%s"%num)
         
    #scene.add_geometry(fem_cuisse, geom_name="tissue_")
    #scene.add_geometry(bg, geom_name="tissue_")
    #scene.add_geometry(tib_mollet, geom_name="tissue_")  

    #if (np.random.random() > 0.25):
    #    gen_surgical_tools(scene, ROOT_DIR + "/objects_test/tools/", np.dot(M_rot_trans, M_flexion))
    #    labels.append(3) #surgical tool
    try:
        fcx_2 = (np.amax(femur.vertices[:,0]) + np.amin(femur.vertices[:,0]))/2
        fcy_2 = (np.amax(femur.vertices[:,1]) + np.amin(femur.vertices[:,1]))/2
        fcz_2 = (np.amax(femur.vertices[:,2]) + np.amin(femur.vertices[:,2]))/2
        fl_2= (np.amax(femur.vertices[:,0])-np.amin(femur.vertices[:,0])) /2
        fw_2= (np.amax(femur.vertices[:,1])-np.amin(femur.vertices[:,1])) /2
        fh_2= (np.amax(femur.vertices[:,2])-np.amin(femur.vertices[:,2])) /2
    except:
        print(num)
        scene.show()
        fcx_2 = 0
        fcy_2 = 0
        fcz_2 = 0
        print("only one object")
    """
    try:
        tcx_2 = (np.amax(tibia.vertices[:,0]) + np.amin(tibia.vertices[:,0]))/2
        tcy_2 = (np.amax(tibia.vertices[:,1]) + np.amin(tibia.vertices[:,1]))/2
        tcz_2 = (np.amax(tibia.vertices[:,2]) + np.amin(tibia.vertices[:,2]))/2
        tl_2= (np.amax(tibia.vertices[:,0])-np.amin(tibia.vertices[:,0])) /2
        tw_2= (np.amax(tibia.vertices[:,1])-np.amin(tibia.vertices[:,1])) /2
        th_2= (np.amax(tibia.vertices[:,2])-np.amin(tibia.vertices[:,2])) /2
    except:
        scene.show()
        tcx_2 = 0
        tcy_2 = 0 
        tcz_2 =0
        print("only one object")
    
    """

    # Write annotation in a txt file for votenet  
    with open(labels_dir+"%06d.txt"%num_scene, 'a') as label_scene:    
        if(type(femur) is trimesh.Trimesh):
            labels.append(1)
        """
            label_scene.write("Femur %f %f %f %f %f %f %f %f %f\n"%(
                            fcx_2, fcy_2, fcz_2,   
                            #femur.extents[0], femur.extents[1], femur.extents[2]   ,                                            #centroid               %f %f %f
                            fl_2,fw_2,fh_2,                                                          #length, width, height  %f %f %f
                            angle_x+flexion, angle_y, angle_z,                              #euler angles           %f %f %f
                            ))
            labels.append(1)

        
        if(type(tibia) is trimesh.Trimesh):
            label_scene.write("Tibia %f %f %f %f %f %f %f %f %f\n"%(
                            tcx_2, tcy_2, tcz_2,                           #centroid               %s (%f %f %f)
                            #tibia.extents[0], tibia.extents[1], tibia.extents[2], 
                            tl_2,tw_2,th_2,                                                           #length, width, height  %f %f %f
                            angle_x, angle_y, angle_z,                                  #euler angles           %f %f %f
                            ))
            labels.append(2)
        """
    label_scene.close()
    np.save(labels_dir+"%06d.npy"%(num_scene), labels)
    return flexion, np.array([fcx_2, fcy_2, fcz_2]),np.array([fcx_2, fcy_2, fcz_2])#np.array([tcx_2, tcy_2, tcz_2])

def gen_surgical_tools(scene, path, transform):
    tools =  [f for f in listdir(path) if isfile(join(path, f))]
    obj = random.choice(tools)
    tool = trimesh.load(path+obj)
    tool.apply_transform(transform)
    scene.add_geometry(tool, geom_name="tool_")


def gen_additional_objects(scene, position):
    p = np.zeros_like(position) -0.05
    p[2]=-0.1  
    M= np.eye(4)       # Translation and Rotation
    M[:3,3]=p
    M[3,:]= [0,0,0,1]
    geom = trimesh.creation.cylinder(radius=0.001+random.random()*0.0015, height= 0.00001, transform=M)
    scene.add_geometry(geom, geom_name="box_")
    #scene.show()

from scipy.spatial.transform import Rotation
def gen_clutter(scene, nb_clutter): 
    
    p = np.empty((nb_clutter, 3))
    if(np.random.random() >0.5):
        choice = pos_tibia
    else:
        choice = pos_femur
    rng = np.random.random((nb_clutter,3))-0.5
    rng[:,0] = rng[:,0] + np.sign(rng[:,0])*0.02
    rng[:,1] = rng[:,1] + np.sign(rng[:,1])*0.02
    p[:,0] = choice[0] + rng[:,0]*0.8
    p[:,1] = choice[0] + rng[:,1]*0.8
    p[:,2] = [choice[2]- (rng[i,2])*0.6
                if(np.abs(rng[i,1]) > 0.25 or np.abs(rng[i,0]> 0.25)) 
                else choice[2]- (rng[i,2]+0.7)*0.4 for i in range(nb_clutter)]

    R = Rotation.random(num=nb_clutter).as_matrix()
    
    M = np.eye(4)       # Translation and Rotation
    M = np.vstack([M]*nb_clutter).reshape((nb_clutter,4,4))
    M[:,:3,:3] = R[:]
    M[:,:3,3] = p[:]
    M[:,3,:] = [0,0,0,1]
    
    for i in range(nb_clutter):
        geom = trimesh.creation.cylinder(radius=np.random.random()*0.3*(np.maximum(0, rng[i,2])+0.5), height= np.random.random()*0.1, transform=M[i])
        geom = trimesh.permutate.noise(geom,np.random.random()*0.035)
        scene.add_geometry(geom, geom_name="clutter_") 


def gen_camera_param():
    resolution_camera=[1024,1024]
    
    cam_trns= np.eye(4,4)
    cam = trimesh.scene.Camera("cam", resolution=resolution_camera, fov=[60,60], z_near=0.05, z_far=500)
    #cam_trns[:3,3]=[0,0,np.random.random()*0.2+0.4]
    
    #with open(calib_dir+"%06d.txt"%i, 'w') as calib_cam:
    #    cam_int = np.eye(3)
    #    cam_ext = scene.camera.K
    #    np.savetxt(calib_cam, (cam_int.reshape(-1), cam_ext.reshape(-1)), fmt='%g')
    #calib_cam.close()
    return cam, cam_trns


def generate(nb_samples, dataset_name):
    print("---------  Generating dataset scenes ---------")
    print("number of scenes: %d"%nb_samples)
    TRAIN_VAL_SPLIT = 0.80
    print("train: %d \t val: %d"%(nb_samples*TRAIN_VAL_SPLIT, np.round(nb_samples*(1-TRAIN_VAL_SPLIT))))
    
    stats = {}; total_flexion = [] ;  total_pixels = [] ; total_femur_dist = [] ; total_tibia_dist = [] ; total_occlusion_femur = [] ; total_occlusion_tibia = []    
    image_final_size=[640,640]
    dataset_path = "../"+dataset_name
    depth_ims_dir, depth_dir, labels_dir, sem_dir, calib_dir= create_target_dirs(dataset_path)
    
    train_dir = ROOT_DIR + "/objects_test/train/"  
    val_dir = ROOT_DIR + "/objects_test/val/"
    train_objects = load_files(train_dir)
    val_objects =  load_files(val_dir)

    camera, camera_transform = gen_camera_param()
    for i in tqdm(range(nb_samples)):        
        scene = init_scene(camera, camera_transform)
        if(i< int(TRAIN_VAL_SPLIT*nb_samples)):
            if(i==0):
                print("--------- Generating train data ---------")
            flexion, pos_femur, pos_tibia = gen_scene(scene, i, train_dir, train_objects, labels_dir)
        else:
            if(i==int(TRAIN_VAL_SPLIT*nb_samples)):
                print("---------- Generating val data ----------")
            flexion, pos_femur, pos_tibia = gen_scene(scene, i, val_dir, val_objects, labels_dir)
        
        for geometry in scene.geometry:
            if "Femur" in geometry or "Tibia" in geometry:
                if geometry.split(':')[0] in stats.keys():
                    stats[geometry.split(':')[0]]+=1
                else:
                    stats[geometry.split(':')[0]]=1
        gen_additional_objects(scene, position=pos_femur)
        pixels = gen_depth_image(depth_dir, depth_ims_dir, scene,i, image_final_size)   
        occlusion_f, occlusion_t = gen_semantic_data(sem_dir, scene, i, image_final_size)
        append_values(total_flexion, flexion); append_values(total_pixels, pixels); append_values(total_occlusion_femur, occlusion_f); append_values(total_occlusion_tibia, occlusion_t)

    compute_mean_values(stats, total_pixels, total_flexion, total_occlusion_tibia, total_occlusion_femur)
    
    write_data_idx(TRAIN_VAL_SPLIT,nb_samples, dataset_path)
    write_stats(dataset_path, stats)

from tqdm import tqdm
import argparse
from datetime import datetime
#TODO: gridsearch, same metrics, base de données bruitée
if __name__ == '__main__':
    time = datetime.today().strftime('%Y-%m-%d-%Hh%M')
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, default=2000)
    parser.add_argument('--name', type= str, default=time)
    FLAGS = parser.parse_args()
    nb_samples = FLAGS.number
    dataset_name = FLAGS.name
    generate(nb_samples, dataset_name)