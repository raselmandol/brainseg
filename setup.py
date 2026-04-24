from setuptools import setup, find_packages

setup(
    name="brainseg-ai",
    version="1.6.5",
    description="Brain abnormality segmentation application",
    author="Md. Rasel Mandol",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "matplotlib",
        "opencv-python",
        "numpy",
        "scipy",
        "psutil",
        "PyQt6",
        "nibabel"
    ],
    entry_points={
        "console_scripts": [
            "brainseg=brainseg.main:main"
        ]
    },
)