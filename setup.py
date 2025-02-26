from setuptools import setup, find_packages
import QuantGplearn

VERSION = QuantGplearn.__version__
setup(
    name='QuantGplearn',
    version=VERSION,
    author='WYFHHH',
    author_email='wangyifei9588@outlook.com',
    description='A systematic framework for factor mining in quantitative investment strategies',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/WYFHHH/QuantGplearn',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    platforms='Linux'
)


