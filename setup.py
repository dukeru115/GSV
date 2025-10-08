"""
Setup script for GSV 2.0 - Global State Vector Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gsv2",
    version="2.0.0",
    author="Timur Urmanov, Kamil Gadeev, Bakhtier Iusupov",
    author_email="urmanov.t@gmail.com",
    description="Global State Vector 2.0: Multi-Scale Control for Autonomous AI Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dukeru115/GSV",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
    },
    py_modules=[
        "gsv2_core",
        "gsv2_qlearning",
        "gsv2_analysis",
        "gsv2_tests",
        "gsv2_gridworld_demo",
        "gsv2_stress_demo",
        "gsv2_experiments",
        "gsv2_quickstart",
    ],
    include_package_data=True,
    keywords="reinforcement-learning, meta-learning, adaptive-systems, stochastic-control, autonomous-agents",
)
