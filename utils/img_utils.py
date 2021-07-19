import numpy as np
from skimage.util import dtype
import trimesh
from skimage import io
import os
import PIL.Image
import PIL.ImageFilter
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

def gen_depth_image(root_dir, path_pc, path_img, scene, scene_number, resolution):

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
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.full(scene.camera.resolution, 0, dtype=np.uint8)
    MAX_DEPTH = 120
        
    depth_float = ((depth - depth.min()) / depth.ptp())

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * MAX_DEPTH).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    a= normalize(a)
    a[a>MAX_DEPTH]=0 #for mask


    if np.random.random() < 0.5:
        background = io.imread(root_dir+'/background_image/'+np.random.choice(os.listdir(root_dir+'/background_image/')))
        background[background>90]=0
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

    mask_array= np.where(np.transpose(a)==0, np.transpose(a), 255)
    mask_img = PIL.Image.fromarray(mask_array).transpose(PIL.Image.FLIP_TOP_BOTTOM)
    
    background_img.paste(depth_map,mask=mask_img)
    background_img = background_img.crop((left,upper,right,lower))

    #background_img = background_img.filter(PIL.ImageFilter.ModeFilter(size=3))
    file_name =path_img+"/%06d.png"%(scene_number)
    background_img.save(file_name)
    return np.asfarray(background_img)

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
        print("%f"%(f), file=label_scene)
        print("%f"%(t), file=label_scene)

    return f,t
