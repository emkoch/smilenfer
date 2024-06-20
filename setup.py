from setuptools import setup

setup(name='smilenfer',
      version='0.1',
      description='Parametric inference from GWAS smiles',
      author='Evan Koch',
      author_email='emkoch@bwh.harvard.edu',
      packages=['smilenfer'],
      install_requires=['pandas', 'numpy', 'scipy', 'numba', 'matplotlib',
                        'seaborn', 'spycial', 'rpy2'])
