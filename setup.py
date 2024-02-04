from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "PYPI_README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.0'
DESCRIPTION = 'Synthetic Data Generation and Evaluation'
LONG_DESCRIPTION = long_description
URL = "https://github.com/SartajBhuvaji"

# Setting up
setup(
    name="sdgne",
    version=VERSION,
    author="Sartaj Bhuvaji",
    author_email="s.bhuvaj@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    package_data={
        '': ['*.csv', '*.h5'],
        'sdgne.demodata': ['*.csv', '*.h5'],  
        'sdgne.models': ['*.h5'],
    },
    keywords=['python','synthetic data', 'autoencoders','smote'],
    url= URL,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    include_package_data=True,
)