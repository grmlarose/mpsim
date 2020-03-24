from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()


requirements = [
    requirement.strip() for requirement in open("requirements.txt").readlines()
]


setup(name="mpsim",
      version="99.9.9",
      description="SImulate (noisy) quantum circuits using matrix product states (MPS).",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="Ryan LaRose",
      author_email="rlarose@google.com",
      url="https://github.com/grmlarose/mps",
      license="MIT",
      python_requires=">=3.6",
      install_requires=requirements,
      packages=find_packages()
      )
