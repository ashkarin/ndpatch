from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ndpatch',
    version='0.0.1',
    description='Extract arbitrary n-dimensional regions from ndarrays.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/ashkarin/ndpatch',
    author='Andrei Shkarin',
    author_email='andrei.shkarin@google.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords='ndarray patch region data development',
    packages=find_packages(exclude=['static']),
    install_requires=['numpy'],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/ashkarin/ndpatch/issues',
        'Source': 'https://github.com/ashkarin/ndpatch'
    }
)