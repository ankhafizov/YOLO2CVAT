from setuptools import find_packages
from distutils.core import setup

setup(
    name="yolo2cvat",
    version="1.0",
    packages=find_packages(),
    long_description="Inference YOLOv5 model to CVAT YOLO1.1 data format",
    include_package_data=True,
    entry_points={
        "console_scripts": ["yolo2cvat=main_yolo2cvat:main"],
    },
    install_requires=[
        "PyYAML==6.0",
        "click",
        "numpy",
        "opencv_python",
        "setuptools",
        "tqdm",
    ],
)
