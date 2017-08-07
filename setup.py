from setuptools import setup
from setuptools import find_packages


setup(
    name='mnist',
    version='0.0.0',
    description='Digit Classification App',
    url='https://github.com/yalotfi/MNIST-Classifier',
    author='Yaseen Lotfi',
    license='MIT',
    packages=find_packages(package_dir='mnist'),
    include_package_data=True,
    install_requires=[
        'flask',
        'numpy',
        'tensorflow',
        'keras',
        'PIL'
    ]
)
