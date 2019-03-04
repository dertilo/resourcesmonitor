from setuptools import setup, find_packages

setup(
    name='resourcesmonitor',
    version='0.1',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license='LICENSE.txt',
    long_description=open('README.md').read(),
    install_requires=[
        'resmon @ git+https://github.com/xybu/python-resmon.git#egg=resmon-1.0.2',
        'matplotlib==3.0.3'
    ],
)