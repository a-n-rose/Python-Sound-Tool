import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

# This call to setup() does all the work
setup(
    name="pysoundtool",
    version="0.1.0a",
    description="A research-based framework for exploring sound as well as machine learning in the context of sound.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/a-n-rose/Python-Sound-Tool",
    author="Aislyn Rose",
    author_email="rose.aislyn.noelle@gmail.com",
    license="AGPL-3.0",
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    packages=find_packages(exclude=("tests","docs","pysoundtool_online", "binder_notebooks", "jupyter_notebooks")),
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    python_requires="==3.6",
)
