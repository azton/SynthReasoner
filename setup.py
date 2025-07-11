#!/usr/bin/env python3
"""
Setup script for LLM Synthetic Reasoning Chain Generator
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-synthetic-reasoner",
    version="1.0.0",
    author="TPC Research Team",
    description="LLM-powered synthetic reasoning chain generator for training reasoning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-synthetic-reasoner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llm-reasoner=run_llm_reasoner:main",
        ],
    },
    keywords="llm reasoning synthetic training data mcq question generation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-synthetic-reasoner/issues",
        "Source": "https://github.com/yourusername/llm-synthetic-reasoner",
    },
)