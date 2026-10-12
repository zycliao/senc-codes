from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="cmake_example",
    version="0.1.0",
    ext_modules=[Pybind11Extension("cmake_example", ["self_inter.cpp"], cxx_std=17)],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
