import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mps",
    version="0.0.1",
    author="Ryan LaRose",
    author_email="rlarose@google.com",
    description="Toolbox for simulating (noisy) quantum circuits with MPS.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux Ubuntu",
    ],
    python_requires='>=3.6',
)
