"""Installable package setup for qml_hybrid."""

from setuptools import setup, find_packages

setup(
    name="qml_hybrid",
    version="0.1.0",
    author="Rajat Sainju",
    author_email="rsainju@anl.gov",
    description=(
        "Hybrid quantum-classical variational circuit classifier "
        "with Bayesian architecture search"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rsainju/qml-hybrid-classifier",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pennylane>=0.38",
        "torch>=2.3",
        "scikit-learn>=1.5",
        "scikit-optimize>=0.10",
        "numpy>=1.26",
        "scipy>=1.13",
        "pandas>=2.2",
        "mlflow>=2.13",
        "matplotlib>=3.9",
        "seaborn>=0.13",
        "pyyaml>=6.0",
        "tqdm>=4.66",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
