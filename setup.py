from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()

with open('development_requirements.txt') as f:
    dev_requirements = f.read().splitlines()


requirements = [
    requirement.strip() for requirement in open("requirements.txt").readlines()
]

description = ("A package for using matrix product states "
               "to simulate quantum circuits.")

setup(name="mpsim",
      version="0.1.0",
      description=description,
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="Ryan LaRose",
      author_email="rlarose@google.com",
      url="https://github.com/grmlarose/mps",
      license="MIT",
      python_requires=">=3.6",
      install_requires=requirements,
      extras_require={
            "development": set(dev_requirements),
            "test": dev_requirements
      },
      packages=find_packages()
      )
