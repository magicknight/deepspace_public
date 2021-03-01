from setuptools import setup, find_packages
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.md')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

version = {}
with open(os.path.join(_here, 'deepspace', 'version.py')) as f:
    exec(f.read(), version)

setup(
    name='deepspace',
    version=version['__version__'],
    description=('DeepSpace is a Artificial Intellegence Framework'),
    long_description='Deep Learning Framework, developed by Zhihua Liang at University of Antwerp. This project was started at Feb 18th, 2020.',
    author='Zhihua Liang',
    author_email='zhihua.liang@uantwerpen.be',
    url='https://github.com/magicknight/deepspace',
    license='MIT',
    packages=find_packages(exclude=("configs", "tests")),
    #   no dependencies in this example
    #   install_requires=[
    #       'dependency==1.2.3',
    #   ],
    #   no scripts in this example
    #   scripts=['bin/a-script'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.8'],
)
