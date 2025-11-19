"""
Setup script for gb_spike_slab package

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="gb_spike_slab",
    version="0.1",
    author="Patrick Meyers",
    author_email="pmeyers@ethz.ch",
    description="A spike and slab approach to galactic binary detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "matplotlib>=3.0",
        "interpax",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="gravitational-waves lisa data-analysis astrophysics",
)
