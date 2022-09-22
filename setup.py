from setuptools import setup, find_packages

#|-fslm
#| |-notebooks
#|	|-setup.py
#|	|-README.md
#|	|-experiment_utils
#|		|-__init__.py
#|		|-__init__.py
#|		|-utils.py
#|		|-metrics.py
#|		|-experiment_helper.py
#|		|-analysis.py
#|		|-toymodels.py
#|	|-fslm
#|		|-snle.py
#|		|-utils.py
#|	|-hh_utils
#|		|-__init__.py
#|		|-utils.py
#|		|-hh_simulator.py
#|		|-extractor.py
#|		|-features.py
#|		|-analysis.py


setup(
   name='fslm',
   version='0.1',
   description='Tools to evaluate feature importance in sbi, specifically for HH models.',
   author='Jonas Beck',
   author_email='jonas.beck@uni-tuebingen.de',
   packages=find_packages(),  # would be the same as name
   install_requires=['sbi', 'seaborn', 'matplotlib', 'pytorch', 'numpy', 'scipy', 'joblib', 'tqdm', 'brian2', 'pandas', 'six', 'cycler'],
)