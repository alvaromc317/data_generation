from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='data_generation',
    version='v0.0.1',
    author='Alvaro Mendez Civieta',
    author_email='almendez@est-econ.uc3m.es',
    license='GNU General Public License',
    zip_safe=False,
    url='https://github.com/alvaromc317/data_generation',
    description='A synthetic data generation package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['synthetic data'],
    python_requires='>=3.5',
    install_requires=["numpy >= 1.15",
                      "scipy >= 1.4.1"],
    packages=find_packages()
)
