import numpy as np
import trimesh

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

def init_scene(camera, camera_transform):
    scene = trimesh.Scene(camera=camera, camera_transform=camera_transform)
    M_wall= np.eye(4,4)
    M_wall[:3,3]=[0,0,np.random.random()*2-3.5]
    wall = trimesh.creation.box([10,10,0.005], M_wall)
    scene.add_geometry(wall, geom_name="wall")
    return scene
