"""
Setup script for claude_gb_toolkit package

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="claude_gb_toolkit",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Modular toolkit for LISA gravitational wave data analysis",
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
