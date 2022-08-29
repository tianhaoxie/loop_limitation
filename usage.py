import torch
from pytorch3d.io import load_objs_as_meshes,save_obj
import loop_limitation
cuda='cuda:0'
cpu='cpu'

path="bunny_314.obj"

m=load_objs_as_meshes([path],device=cuda)
lp = loop_limitation.loop_limitation()

v=m.verts_packed()
f=m.faces_packed()

#initialize the Jacobian matrix
lp.init_J(v,f)
#get Jacobian matrix
J = lp.get_J()

#compute limit position
l = lp.compute_limitation(v)

save_obj("test.obj",l,f)




