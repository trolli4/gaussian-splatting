what is _C?

## diff_gaussian_rasterization
### setup.py
summarizes multiple files into one extension?

### ext.cpp
pybind; makes C++ files from 'rasterize_points.h' usable in python

### __init__.py
#### _RasterizeGaussians.forward()
formats given arguments in a way that C++ can work with them
_RasterizeGaussians.forward() calls '_C.rasterizeGaussians(*args)' which in turn calls '_RasterizeGaussians.apply(< other args >)'

## scene
### gaussian_model
