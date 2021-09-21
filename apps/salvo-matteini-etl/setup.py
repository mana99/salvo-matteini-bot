
from setuptools import find_namespace_packages, setup
from os.path import abspath, dirname, join

def readme():
    with open('README.md') as f:
        return f.read()


def requirements(fn):
    with open(fn) as f:
        return [line.strip() for line in f.readlines() if not line.startswith("--")]


setup(name="salvo-matteini-etl",
      version="0.1.0",
      description="Salvo Matteini ETL",
      long_description=readme(),
      package_dir={'': 'src'},
      packages=find_namespace_packages(where='src'),
      install_requires=requirements('requirements.txt'),
      include_package_data=True,
      entry_points={
            'console_scripts': [
                  'salvo-matteini-etl = salvo_matteini_etl.console:main',
            ]
      }
      )
