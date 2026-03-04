import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = []
this_dir = os.path.dirname(__file__)

# build directly from sources
sources = [os.path.join(this_dir, "bindings.cpp")]
cuda_src = [os.path.join(this_dir, "kernels", "vector_add.cu")]
sources.extend(cuda_src)
ext_modules = [
  CUDAExtension(
    "miniinfer_ext._ext",
    sources=sources,
    include_dirs=[os.path.join(this_dir, "kernels")],
  )
]


setup(
  name="miniinfer_ext",
  version="0.1.0",
  packages=["miniinfer_ext"],
  package_dir={"miniinfer_ext": "miniinfer_ext"},  # package located place
  ext_modules=ext_modules,
  cmdclass={"build_ext": BuildExtension},
)
