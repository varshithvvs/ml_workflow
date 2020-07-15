# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ml_workflow',
    version='0.0.1',
    description='A Data Science project for office hackathon',
    long_description=readme,
    author='Sai Varshith VV',
    author_email='svvarsham@gmail.com',
    url='https://github.com/varshithvvs/ml_workflow',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

