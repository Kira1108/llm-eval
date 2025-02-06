from setuptools import setup
from setuptools import find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()
    
# requires = parse_requirements('requirements.txt')

setup(name='easyscore',
      version='0.0.1',
      description='fucking llm evaluation',
      author='The fastest man alive.',
      packages=find_packages(),
      install_requires=[])