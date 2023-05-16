from setuptools import setup, find_packages


setup(name='mockFRBhosts',
      version='0.1',
      description='Package to simulate optical FRB follow-up',
      author='Joscha N. Jahns',
      author_email='jjahns@mpifr-bonn.mpg.de',
      url='https://github.com/JoschaJ/mockFRBhosts',
      packages=find_packages(),
      license='tbd',
      install_requires=[
          'python>=3.6',  # because of f-strings
          'numpy',
          'scipy',
          'pandas',
          'matplotlib',
          'seaborn',
          'astropy',
          ]
      )
