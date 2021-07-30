import numpy as np
import trimesh
from skimage import io
import os
import PIL.Image
import PIL.ImageFilter
import PIL.ImageEnhance

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    newMin = 0
    newMax = 255 
    """
    new_arr=np.zeros_like(arr)
    minval = np.min(arr)
    maxval = np.max(arr)
    new_arr[:,:] = ((arr[:,:]-minval)/(maxval-minval))*255

    arr = new_arr.astype(np.uint8)
    return arr


MAX_DEPTH = 100
def gen_depth_image(path_img, scene, scene_number, resolution, bg_dir, train=False):

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    dump = scene.dump(concatenate=True)
    # do the actual ray- mesh queries
    pye = trimesh.ray.ray_pyembree.RayMeshIntersector(dump)
    points, index_ray, _ = pye.intersects_location(
        origins, vectors, multiple_hits=False)

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[0],
                                      vectors[index_ray])
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.full(scene.camera.resolution, 0, dtype=np.uint8) 

    # convert depth into 0 - 255 uint8   
    depth_float = ((depth - depth.min()) / depth.ptp())
    depth_int = (depth_float * MAX_DEPTH).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    a= normalize(a)
    a[a>MAX_DEPTH]=0 # all background pixels are set to black

    if np.random.random() < 0.5:
        background = io.imread(bg_dir+np.random.choice(os.listdir(bg_dir)))
        background_img = PIL.Image.fromarray(background)
        background_img = background_img.rotate(np.random.random()*360)
    else: 
        x= (np.random.rand(1024,1024)-0.5)*40+np.median(a[a>0])
        background = np.array(x, dtype = np.uint8)
        background_img = PIL.Image.fromarray(background)
        background_img = background_img.filter(PIL.ImageFilter.GaussianBlur(radius=20))

    # create a PIL image from the depth queries
    depth_map = PIL.Image.fromarray(np.transpose(a))
    depth_map = depth_map.transpose(PIL.Image.FLIP_TOP_BOTTOM) #flip to be in the "camera view"
    bg_w, bg_h = background_img.size
    w, h = resolution

    left = (bg_w-w)//2
    upper = (bg_h-h)//2
    right = bg_w - (bg_w-w)//2
    lower = bg_h - (bg_h-h)//2

    #all black pixels are masked, only objects are kept
    mask_array= np.where(np.transpose(a)==0, np.transpose(a), 255)
    mask_img = PIL.Image.fromarray(mask_array).transpose(PIL.Image.FLIP_TOP_BOTTOM)
    
    background_img.paste(depth_map,mask=mask_img)
    background_img = background_img.crop((left,upper,right,lower))

    if(train):
        file_name =path_img+"/train/%06d.png"%(scene_number)
    else:
        file_name =path_img+"/val/%06d.png"%(scene_number)
    w,h= background_img.size
    
    background_img.save(file_name)
    return np.asfarray(background_img), points

TOOL = 3
TIBIA = 2
FEMUR = 1
BG = 0
def gen_semantic_data(path, scene, scene_number, labels_dir, resolution):
    labels= []
    origins, vectors, pixels = scene.camera_rays()

    copy_node = scene.geometry.copy()
    arr_geom = []
    semantic = np.zeros(scene.camera.resolution)
    depth_map = np.full(scene.camera.resolution,255)
    px_cnt = {}
    i=1 #0 is for background
    for geometry in reversed(copy_node):
        if(geometry=="wall"):
            continue
        copy_scene = scene.copy()
        for other in copy_node:
            if (other != geometry):
                copy_scene.delete_geometry(other)
        dump = copy_scene.dump(concatenate=True)
        pye = trimesh.ray.ray_pyembree.RayMeshIntersector(dump)
        points, index_ray, _ = pye.intersects_location(origins, vectors, multiple_hits=False)
 
        # for each hit, find the distance along its vector
        depth = trimesh.util.diagonal_dot(points - origins[0],
                                        vectors[index_ray])

        # find pixel locations of actual hits
        pixel_ray = pixels[index_ray]

        #if object (depth) is closer than the current depth map, the semantic map takes its value, else we keep the original value
        semantic[pixel_ray[:,0], pixel_ray[:,1]] = np.array([i  if b <= a  else o for (a, b, o) in zip(    
                                                                                                    depth_map[pixel_ray[:,0],pixel_ray[:,1]], 
                                                                                                    depth, 
                                                                                                    semantic[pixel_ray[:,0], pixel_ray[:,1]])]
                                                                                                    )
        #keep the closest object in memory
        depth_map[pixel_ray[:,0], pixel_ray[:,1]] = np.array([min(a, b) for (a, b) in zip(depth_map[pixel_ray[:,0],pixel_ray[:,1]], depth)])

        if("Femur" in geometry or "Tibia" in geometry or "Guide" in geometry):
            name=geometry.split('_')[0]
            labels.append((i,name))
            px_cnt[name]=len(pixel_ray)
            arr_geom.append(geometry)
        i+=1

    # create a PIL image from the depth queries
    semantic_map = PIL.Image.fromarray(np.transpose(semantic))  
    semantic_map = semantic_map.transpose(PIL.Image.FLIP_TOP_BOTTOM)
   
    background = PIL.Image.fromarray(np.zeros((1024,1024)),mode='L')
    bg_w, bg_h = background.size
    background.paste(semantic_map)
    w, h = resolution
    left = (bg_w-w)//2
    upper = (bg_h-h)//2
    right = bg_w - (bg_w-w)//2
    lower = bg_h - (bg_h-h)//2
    background = background.crop((left,upper,right,lower))
    file_name = path+"%06d.png"%(scene_number)
    background.save(file_name)
    np.save(labels_dir+"%06d.npy"%(scene_number), labels)

    t,f,g = 0, 0, 0 # to be used later for occlusion computation
    return f,t,g
