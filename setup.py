from setuptools import setup
from setuptools import find_packages
import pathlib

# Read the contents of your README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='hybrid-platform',
    version='0.0.2',
    description='Platform domain Gymnasium environment',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Craig Bester',
    packages=find_packages(),
    package_data={
        'gym_platform': ['envs/assets/*.png']
    },
    install_requires=[
        'gymnasium',
        'pygame>=1.9.3',
        'numpy>=1.14.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
) 
