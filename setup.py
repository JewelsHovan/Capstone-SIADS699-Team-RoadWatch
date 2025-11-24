#!/usr/bin/env python3
"""
Setup script for Texas Crash Prediction project

Install in development mode:
    pip install -e .

This allows importing from anywhere:
    from config.paths import BRONZE, SILVER, GOLD
    from data_engineering.utils.validation import check_for_leakage
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
with open(requirements_file) as f:
    # Filter out duplicates and empty lines
    requirements = []
    seen = set()
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            # Extract package name (before ==, >=, etc.)
            pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].strip()
            if pkg_name not in seen:
                requirements.append(line)
                seen.add(pkg_name)

# Read ML requirements
ml_requirements_file = Path(__file__).parent / "requirements_ml.txt"
if ml_requirements_file.exists():
    with open(ml_requirements_file) as f:
        ml_requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                pkg_name = line.split('==')[0].split('>=')[0].strip()
                if pkg_name not in seen:
                    requirements.append(line)
                    seen.add(pkg_name)

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
with open(readme_file, encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="texas-crash-prediction",
    version="1.0.0",
    author="Julien Hovan",
    author_email="jhovan@umich.edu",
    description="Machine learning system for predicting crash severity in Texas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JewelsHovan/crash-prediction",
    packages=find_packages(exclude=['tests', 'notebooks', 'docs', 'deepthi-src']),
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'crash-pipeline=scripts.run_pipeline:main',
            'crash-train=ml_engineering.train_with_mlflow:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
