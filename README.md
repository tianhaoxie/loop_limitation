# loop_limitation

A pytorch cpp extension for evaluating loop limit subdivision surface which was used in

https://arxiv.org/abs/2208.01685
and
https://arxiv.org/abs/2203.13333

## Install

```
git clone --recurse-submodules https://github.com/tianhaoxie/loop_limitation.git
pip install -r requirements.txt
pip install .
```

## Usage
```
import torch
import loop_limitation
```
This lib should be imported after torch, otherwise Python will report errors.
Then this lib can be intialized by the line below.
```
lp = loop_limitation.loop_limitation()
```
After the mesh file was loaded by any lib, such as Pytorch3D in our example, the constant Jacobian matrix of a certain mesh
can be computed by
```
lp.init_J(v,f)
```
where the v is the vertices, and f is the faces. To be noted that the v and f must be torch tensor, but can be either on CUDA or CPU. The result 
will be on the same device as the input.
The computed Jacobian matrix can be got by
```
J = lp.get_J()
```
The device is same as the input.
```
l = lp.compute_limitation(v)
```
This function returns the limit position of the input vertices, which was evaluated by the method based on Jos Stam's algorithm[1,2].



## Reference
[1] Stam, J. (1998, July). Evaluation of loop subdivision surfaces. In SIGGRAPH’98 CDROM Proceedings.

[2] Xie, T. (2022). Differentiable Subdivision Surface Fitting. arXiv preprint arXiv:2208.01685.


