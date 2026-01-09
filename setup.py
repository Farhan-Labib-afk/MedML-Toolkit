from pathlib import Path

from setuptools import find_packages, setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="medml-toolkit",
    version="0.1.0",
    author="Farhan Labib",
    description="MedML Toolkit - A toolkit for predictive modeling and feature analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
        "streamlit>=1.33.0",
    ],
)
