# Questions 
- what is _C?
- what is 'BasicPointCloud'?
- what i want to change probably are 
  - ``GaussianModel.densify_XX()`` functions
  - ``forward.cu`` 
  - ``backward.cu``
- what is [sparse adam](https://youtu.be/MD2fYip6QsQ?si=SMl8A8t1ycWLg237)?

<br> <br>

# Project Overview

## gaussian_renderer/\_\_init__.py
- take closer look
- renders the scene for camera at ``viewpoint_camera``
- ``render(.....)["render"]`` is image seen from camera ``viewpoint_camera``

## submodules/diff_gaussian_rasterization
### setup.py
- summarizes multiple files into one extension?
- this extension can then be used like any external module using ```#import <extension>```

### ext.cpp
- pybind; makes C++ files from 'rasterize_points.h' usable in python

### \_\_init__.py

#### _RasterizeGaussians(torch.autograd.Function)
- subclass of ```torch.autograd.Function```
- custom autograd function
- [explanation of what a (custom) autograd function is](https://brsoff.github.io/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html)

#### _RasterizeGaussians.forward()
- computes output for given input ?
- rasterizes gaussians
- formats given arguments in a way that C++ can work with them
```_RasterizeGaussians.forward()``` calls ```_C.rasterizeGaussians(*args)``` >>> ```_C.rasterizeGaussians(..)``` is CUDA rasterizer which is used inside the function

#### _RasterizerGaussian.backward()
- computes input for given output ?
- calculates gradients of gaussians/tensors, returns tuple 'grad' which has same variables as 'rasterize_gaussians()' takes
- formats arguments for C++ lib
- calls '_C.rasterize_gaussians_backward(*args)'

#### GaussianRasterizationSettings
- essentially just a tuple

#### GaussianRasterizer.markvisible()
- calculates which gaussians (or rather central points of which gaussians) are visible (from camera?)

#### GaussianRasterizer.forward()
- does some checks if prerequisites are met, if so calls 'rasterize_gaussians'

### cuda_rasterizer/forward.cu
- [``forward.cu`` vs ``backward.cu``](https://sandokim.github.io/cuda/cuda-rasterizer-foward-cu-backward-cu/)
- tile-based rendering
- rasterization around line 360

#### computeColorFromSH()

#### computeCov2D()

#### computeCov3D()

#### render()

#### preprocess()

### cuda_rasterizer/backward.cu
- backpropagation?
- positional Gradient (line 625 ff.)

#### computeColorFromSH()

#### computeCov2DCUDA()

#### computeCov3D()

#### preprocess()

#### render()

<br>

## submodules/fused_ssim
- compares ground truth with current gaussian splatting state

<br>

## submodules/simple_knn
- knn = "künstliches neuronales Netz"; underlying neural net

<br>

## scene
### __init__
- maybe modify `save()` to store camera views of scene

### gaussian_model
- imports torch.nn; Neural Nets?

#### create_from_pcd()
- creates initial Gaussians from Point Cloud 

#### training_setup()

#### update_learning_rate()

#### construct_list_of_attributes()

#### save_ply()
- investigate for understanding the viewer

#### reset_opacity()

#### load_ply()

#### replace_tensor_to_optimizer() ??

#### _prune_optimizer()

#### prune_points()

#### cat_tensors_to_optimizer()

#### densification_postfix()  
- called by ```densify_and_split()```

#### densify_and_split()
- takes gradients and gradient_threshhold
- N=2, .repeat(2,1) >> split into two Gaussians?

#### densify_and_clone()

#### densify_and_prune()

#### add_densification_stats()

## train.py
- calculates camera view (around line 111, ff.)