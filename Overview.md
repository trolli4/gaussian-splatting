what is _C?

what i want to change probably are only forward.cu and backward.cu?

## diff_gaussian_rasterization
### setup.py
summarizes multiple files into one extension?

### ext.cpp
pybind; makes C++ files from 'rasterize_points.h' usable in python

### __init__.py
#### _RasterizeGaussians.forward()
rasterizes gaussians

formats given arguments in a way that C++ can work with them
_RasterizeGaussians.forward() calls '_C.rasterizeGaussians(*args)' which in turn calls '_RasterizeGaussians.apply(< other args >)'? > '_C.rasterizeGaussians(..)' is CUDA rasterizer which is used inside the function

#### _RasterizerGaussian.backward()
calculates gradients of gaussians/tensors, returns tuple 'grad' which has same variables as 'rasterize_gaussians()' takes

formats arguments for C++ lib
calls '_C.rasterize_gaussians_backward(*args)'

#### GaussianRasterizationSettings
essentially just a tuple

#### Gaussianrasterizer.markvisible()
calculates which gaussians (or rather central points of which gaussians) are visible (from camera?)

#### Gaussianrasterizer.forward()
does some checks if prerequisites are met, if so calls 'rasterize_gaussians'

## scene
### gaussian_model
imports torch.nn; Neural Nets?

#### GaussianModel.create_from_pcd()
what is 'BasicPointCloud'?