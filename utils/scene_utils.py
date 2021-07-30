import numpy as np
import trimesh
from utils.dir_utils import load_files

def init_scene(camera, camera_transform, far):
    # generate a wall in the background for a proper depth computation
    scene = trimesh.Scene(camera=camera, camera_transform=camera_transform)
    position = np.min(far)+np.random.random()*(np.max(far)-np.min(far))
    M_wall= np.eye(4,4)
    M_wall[:3,3]=[0,0, -position]
    wall = trimesh.creation.box([10,10,0.005], M_wall)
    scene.add_geometry(wall, geom_name="wall")
    return scene

def gen_tissue():
    # Not used for now
    fem_cuisse= trimesh.creation.cylinder(radius=0.07, height=0.25, sections=64)
    cut = trimesh.creation.capsule(height=0.4, radius=0.077, count=[64, 64])
    M_cuisse= np.eye(4)       # Translation and Rotation
    M_cuisse[:3,:3]= rotx(np.pi/6)
    M_cuisse[:3,3]= np.array([0,0.025,0.07])
    M_cuisse[3,:]= [0,0,0,1]
    cut.apply_transform(M_cuisse)
    fem_cuisse = fem_cuisse.difference(cut, engine="blender")
    scale= [[   0.95,   0,      0,      0],
            [   0,      0.85,   0,      0],
            [   0,      0,      0.9,    0],
            [   0,      0,      0,      1]]
    fem_cuisse.apply_transform(scale)
    return fem_cuisse


def gen_scene(scene, dir_objects, list_objects, objects, dist_objects, v1=True):
    obj = np.random.choice(list_objects)
    num = obj.split('.')[0].split('_')[1]
    ###################### global transform
    angle_x = 0 #(np.random.random()*np.pi/3)# - np.pi/6 # -18 ~ +18 degree
    angle_y = (np.random.random()*9*np.pi/10) - 9*np.pi/20
    angle_z = (np.random.random()*np.pi) - np.pi/2
    Rx = rotx(angle_x)
    Ry = roty(angle_y)
    Rz = rotz(angle_z)
    R = np.dot(Rz, np.dot(Ry,Rx))  
    t=np.array([(np.random.random()-0.5)*0.1,(np.random.random()-0.5)*0.1,0]) 

    t[2]+=np.min(dist_objects)+np.random.random()*(np.max(dist_objects)-np.min(dist_objects))
    M_rot_trans= np.eye(4)       # Translation and Rotation
    M_rot_trans[:3,:3]= R
    M_rot_trans[:3,3]=-t
    M_rot_trans[3,:]= [0,0,0,1]

    # For each bone, we create a cylinder that imitates muscular tissues 
    # and do the boolean difference between the bone and the cylinder.
    # The CAD models aren't watertight (closed) so the properties "centroid" 
    # of trimesh doesn't give the exact centroid, we compute the error ourselves.
    if(objects['Femur']):
        femur = trimesh.load(dir_objects+'Femur_%s.stl'%num)
        _, fw, fh = femur.extents/2
        fcx, fcy, fcz = np.mean(femur.bounds, axis=0)
        err_f = np.array([fcx,fcy,fcz])-femur.centroid
        flexion = -np.pi/3-(np.random.random()-0.5)*np.pi/2    # 15° -- 105°
        R_femur = rotx(flexion)
        M_flexion= np.eye(4)       # Translation and Rotation
        M_flexion[:3,:3]= R_femur
        M_flexion[3,:]= [0,0,0,1]  
        femur.apply_transform(M_flexion)
        femur.apply_transform(M_rot_trans)
        fcx, fcy, fcz = femur.centroid+np.dot(R, np.dot(R_femur,err_f))
        vec = np.dot(np.dot(R,np.dot(R_femur,rotx(-np.pi/2))), np.array([0,0, fh]))
        if(v1):
            #v1, simple method, robust
            fem_cuisse= trimesh.creation.cylinder(radius=1.3*fw, height=2*fh)
            M_cuisse= np.eye(4)       # Translation and Rotation
            M_cuisse[:3,:3]= np.dot(R,np.dot(R_femur,rotx(-np.pi/2) ))
            M_cuisse[:3,3]= np.array([fcx,fcy,fcz]) + vec
            M_cuisse[3,:]= [0,0,0,1]
            fem_cuisse.apply_transform(M_cuisse)
            femur = femur.difference(fem_cuisse, engine="blender")
        else:
            #v2, complex shape, robustness not ensured
            fem_cuisse = trimesh.load('cuissev2.stl')
            M_cuisse= np.eye(4)       # Translation and Rotation
            M_cuisse[:3,:3]= np.dot(R,np.dot(R_femur,rotx(np.pi/2)))
            M_cuisse[:3,3]= np.array([fcx,fcy,fcz])  -vec
            M_cuisse[3,:]= [0,0,0,1]
            fem_cuisse.apply_transform(M_cuisse)
        if(objects['Tissue']):
            scene.add_geometry(fem_cuisse, geom_name="tissue_")
        scene.add_geometry(femur, geom_name="Femur_%s"%num)
        
        _, fw, fh = femur.extents/2
        fcx, fcy, fcz = np.mean(femur.bounds, axis=0)
        positions = np.array([fcx, fcy, fcz])
        if (objects['Tool'] and np.random.random() > 0.25):
            gen_surgical_tools(scene, "../objects/tools/", np.dot(M_rot_trans, M_flexion), False)  
    
    
    if(objects['Tibia']):
        tibia = trimesh.load(dir_objects+'Tibia_%s.stl'%num)
        _, tw, th = tibia.extents/2
        tcx, tcy, tcz = np.mean(tibia.bounds, axis=0)
        err_t = np.array([tcx,tcy,tcz])-tibia.centroid
        tibia.apply_transform(M_rot_trans)
        tcx, tcy, tcz = tibia.centroid+np.dot(R, err_t)
        tib_mollet= trimesh.creation.cylinder(radius=tw*1.4, height=1.8*th, transform=[[1,0,0,0],[0,1,0,0],[0,0,1,th],[0,0,0,1]])
        M_mollet= np.eye(4)       # Translation and Rotation
        M_mollet[:3,:3]= np.dot(R,rotx(np.pi/2))
        M_mollet[:3,3]= np.array([tcx,tcy,tcz])
        M_mollet[3,:]= [0,0,0,1]
        tib_mollet.apply_transform(M_mollet)
        tibia = tibia.difference(tib_mollet, engine="blender") 
        scene.add_geometry(tibia, geom_name="Tibia_%s"%num)
        if(objects['Tissue']):
            scene.add_geometry(tib_mollet, geom_name="tissue_")

    return flexion, positions


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
    """ Given three Euler angles, return the corresponding rotation matrix in the order X,Y,Z"""
    rotmat = np.zeros((3,3))
    X = rotx(Rx)
    Y = roty(Ry)
    Z = rotz(Rz)
    rotmat = np.dot(Z, np.dot(Y,X))
    return rotmat

def gen_additional_objects(scene, position, near):
    p = np.zeros_like(position) -0.05
    p[2]= np.min(near)+np.random.random()*(np.max(near)-np.min(near))
    M= np.eye(4)       # Translation and Rotation
    M[:3,3]=-p
    M[3,:]= [0,0,0,1]
    geom = trimesh.creation.cylinder(radius=0.001+np.random.random()*0.0015, height= 0.00001, transform=M)
    scene.add_geometry(geom, geom_name="box_")

def gen_surgical_tools(scene, path, transform, first_load):
    tools =  load_files(path, first_load)
    obj = np.random.choice(tools)
    tool = trimesh.load(path+obj)
    scale = np.eye(4)*0.9
    tool.apply_transform(scale)
    tool.apply_transform(transform)
    scene.add_geometry(tool, geom_name="Guide")
