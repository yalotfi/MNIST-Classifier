from setuptools import setup


setup(
    name='mnist',
    packages=[
        'mnist',
        'classifier'
    ],
    include_package_data=True,
    install_requires=[
        'flask'
    ]
)
