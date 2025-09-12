from setuptools import setup, find_packages

setup(
    name='aipmt',
    version='0.1.0',
    author='Farhan Labib',
    description='AI Precision Medicine Training - A toolkit for predictive modeling and feature analysis',
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
        'tqdm>=4.60.0',
        'scipy>=1.7.0'
    ],
)