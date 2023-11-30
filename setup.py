from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="ciropt",
    version="0.0.1",
    packages=["ciropt"],
    license="GPLv3",
    description="Fitting Multilevel Low Rank Matrices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
