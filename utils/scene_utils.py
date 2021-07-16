import numpy as np
import trimesh
from utils.dir_utils import load_files

def init_scene(camera, camera_transform):
    scene = trimesh.Scene(camera=camera, camera_transform=camera_transform)
    M_wall= np.eye(4,4)
    M_wall[:3,3]=[0,0,np.random.random()*2-3.5]
    wall = trimesh.creation.box([10,10,0.005], M_wall)
    scene.add_geometry(wall, geom_name="wall")
    return scene

def gen_scene(scene, num_scene, dir_objects, list_objects, labels_dir):
    labels=[]

    obj = np.random.choice(list_objects)
    num = obj.split('.')[0].split('_')[1]
    if "13" in num:
        num="027"
    femur = trimesh.load(dir_objects+'Femur_%s.stl'%num)
    tibia = trimesh.load(dir_objects+'Tibia_%s.stl'%num)
    ####################### store half-extents
    fl, fw, fh = femur.extents/2
    tl, tw, th = tibia.extents/2
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

    ####################### updaate half-extents
    fl, fw, fh = femur.extents/2
    tl, tw, th = tibia.extents/2
    ####################### update GT centroids post transform
    fcx, fcy, fcz = np.mean(femur.bounds, axis=0)
    tcx, tcy, tcz = np.mean(tibia.bounds, axis=0)

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
    return flexion, np.array([fcx, fcy, fcz]), np.array([fcx, fcy, fcz])#np.array([tcx_2, tcy_2, tcz_2])


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

def gen_additional_objects(scene, position):
    p = np.zeros_like(position) -0.05
    p[2]=-0.1  
    M= np.eye(4)       # Translation and Rotation
    M[:3,3]=p
    M[3,:]= [0,0,0,1]
    geom = trimesh.creation.cylinder(radius=0.001+np.random.random()*0.0015, height= 0.00001, transform=M)
    scene.add_geometry(geom, geom_name="box_")

def gen_surgical_tools(scene, path, transform):
    tools =  load_files(path)
    obj = np.random.choice(tools)
    tool = trimesh.load(path+obj)
    tool.apply_transform(transform)
    scene.add_geometry(tool, geom_name="tool_")


from scipy.spatial.transform import Rotation
def gen_clutter(scene, nb_clutter, position): 
    p = np.empty((nb_clutter, 3))
    rng = np.random.random((nb_clutter,3))-0.5
    rng[:,0] = rng[:,0] + np.sign(rng[:,0])*0.02
    rng[:,1] = rng[:,1] + np.sign(rng[:,1])*0.02
    p[:,0] = position[0] + rng[:,0]*0.8
    p[:,1] = position[0] + rng[:,1]*0.8
    p[:,2] = [position[2]- (rng[i,2])*0.6
                if(np.abs(rng[i,1]) > 0.25 or np.abs(rng[i,0]> 0.25)) 
                else position[2]- (rng[i,2]+0.7)*0.4 for i in range(nb_clutter)]

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
