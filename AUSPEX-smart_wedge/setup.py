# -*- coding: utf-8 -*-
"""
Docstrings do projeto.

"""
from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_file = f.read()

setup(
    name='AUSPEX',
    version='1.5.0',
    description='',
    long_description=readme,
    author='Equipe AUSPEX-UTFPR',
    author_email='danielpipa@utfpr.edu.br',
    url='https://github.com/danielpipa/AUSPEX',
    license=license_file,
    packages=find_packages(exclude=('tests', 'documentation'))
)
