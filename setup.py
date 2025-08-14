from setuptools import setup, find_packages
import os

def _all_data_files():
    pkg = "smilenfer"
    data_root = os.path.join(pkg, "data")
    out = []
    for root, _, files in os.walk(data_root):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, start=pkg)
            out.append(rel)
    return out

setup(name='smilenfer',
      version='0.1',
      description='Parametric inference from GWAS smiles',
      author='Evan Koch',
      author_email='evan.koch@yale.edu',
      packages=find_packages(include=["smilenfer", "smilenfer.*"]),
      include_package_data=True,
      package_data={'smilenfer': _all_data_files()},
      install_requires=['pandas', 'numpy', 'scipy', 'numba', 'matplotlib',
                        'seaborn', 'spycial', 'rpy2'])
