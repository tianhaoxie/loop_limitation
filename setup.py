import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name='loop_limitation',
    version='0.0.1',
    author='Tianhao Xie',
    description='A pytorch cpp extension for evaluating loop limit subdivision surface Resources',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    py_modules=["loop_limitation"],
    install_requires=["torch"],
    ext_modules=[
        CppExtension(
            name='loop_limitation',
            sources=[
                './src/loop.cpp',
                './src/loopEvaluation/eigenStructure.cpp',
                './src/loopEvaluation/limitationSurfaceComputation.cpp',
                './src/loopEvaluation/LoopData.cpp',
                './src/loopEvaluation/pointCollect.cpp',
                './src/loopEvaluation/pointEvaluate.cpp',
            ],
            include_dirs=[os.path.abspath('./thirdparty/eigen'), os.path.abspath('./thirdparty/libigl/include'), os.path.abspath('./src/loopEvaluation')],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
