from setuptools import setup, find_packages

setup(
    name="brainseg",
    version="1.1.0",
    description="Brain abnormality segmentation application",
    author="Md. Rasel Mandol",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "gradio",
        "opencv-python",
        "numpy",
        "scipy",
        "PyQt6"
    ],
    entry_points={
        "console_scripts": [
            "brainseg=brainseg.main:main"
        ]
    },
)
