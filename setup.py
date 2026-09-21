from setuptools import setup, find_packages

setup(
    name="sparc-benchmark",
    version="1.0.0",
    author="T. Abram",
    author_email="tabram@gth-physics.org",
    description="Zero-Free-Parameter SPARC 175-Galaxy Benchmark Suite",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.9",
)
