from setuptools import setup


setup(
    name='mnist-app',
    packages=[
        'app',
        'classifier'
    ],
    include_package_data=True,
    install_requires=[
        'flask'
    ]
)
