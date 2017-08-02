from setuptools import setup


setup(
    name='mnist-app',
    packages=['mnist-app'],
    include_package_data=True,
    install_requires=[
        'flask'
    ],
)
