from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "tensorflow==2.9.3",
    "numpy==1.13.3",
]

setup(
    name='linear',
    version='0.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Classifier test'
)
