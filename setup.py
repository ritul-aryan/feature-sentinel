from setuptools import setup, find_packages

setup(
    name="feature-sentinel",
    version="1.0.0",
    author="Ritul Aryan",
    description="Statistical telemetry gateway & ingestion circuit breaker for ML feature stores",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ritul-aryan/feature-sentinel",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)