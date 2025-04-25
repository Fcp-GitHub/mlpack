from setuptools import setup, find_packages

setup(
    name='mlpack',                              # Name used to import the package
    version='0.1.0',                            # Version of the package
    packages=find_packages(include=['mlpack'])  # Tell `setuptools` to find all packages within the `mlpack` directory
)
