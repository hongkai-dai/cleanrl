from setuptools import setup, find_packages

# Read the contents of requirement file
with open("requirements/requirements.txt") as f:
    requirements = f.read().splitlines()
   
setup(
    name="cleanrl",
    packages=find_packages(),
    install_requires=requirements,
)
