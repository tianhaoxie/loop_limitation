import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='loop_limitation',
    version='0.0.1',
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