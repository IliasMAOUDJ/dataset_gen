import trimesh
import numpy as np

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

cuisse = trimesh.creation.cylinder(0.2,1)

tibia = trimesh.creation.capsule(height=1, radius=1, count=[32, 32])
tibia.show()
M_cuisse= np.eye(4)       # Translation and Rotation
M_cuisse[:3,3]= np.array([0,0.05,0.5])
M_cuisse[3,:]= [0,0,1,1]
tibia.apply_transform(M_cuisse)
cuisse = cuisse.difference(tibia, engine="blender")


#cuisse_int = trimesh.creation.cylinder(0.18,1)
#cuisse = cuisse.difference(cuisse_int, engine="blender")

cuisse.show()
cuisse.export("cuisse_spher.stl")
