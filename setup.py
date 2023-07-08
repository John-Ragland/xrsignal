from setuptools import setup

setup(
    name='xrsignal',
    version='0.0.0',
    author='John Ragland',
    author_email='jhrag@uw.edu',
    description='A python package that ports many of the scipy.signal functions to xarray and distributed computing',
    packages=['xrsignal'],
    install_requires=[
        'xarray'
        'scipy',
    ],
)
