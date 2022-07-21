from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='torchrec_mapper',
      version='0.0.0',
      packages=find_packages(exclude=["tests"]),
      requires=['torch', 'torchrec'],
      ext_modules=[
          cpp_extension.CUDAExtension(
              'torchrec_mapper_cpp',
              [
                  'src/bind.cpp',
              ],
              extra_compile_args=['-std=c++17', '-O2', '-I/usr/local/include'],
          )
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
