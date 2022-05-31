import setuptools
import re

# c.f. https://packaging.python.org/tutorials/packaging-projects/

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

VERSIONFILE = "pdcleaner/__version__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version_str = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setuptools.setup(
    name="pdcleaner",
    version=version_str,
    author="Renan Hilbert",
    author_email="renan.hilbert@eurodecision.com",
    description="A pandas extension for cleaning datasets.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://edgitlab.eurodecision.com/data/pandas-cleaner",
    project_urls={
        "Documentation":
        "https://edgitlab.eurodecision.com/data/pandas-cleaner/-/blob/dev/README.rst",
        "Source Code":
        "https://edgitlab.eurodecision.com/data/pandas-cleaner",
        "Bug Tracker":
        "https://edgitlab.eurodecision.com/data/pandas-cleaner/-/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
    #package_dir={"pdcleaner": "pdcleaner"},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
    setup_requires=["wheel"],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "requests",
        ],
)
