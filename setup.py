from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="biopredict",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="NHANES Data Analysis Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/biopredict",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.0",
        "mlflow>=1.30.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.10.0",
        "flake8>=5.0.0",
        "mypy>=0.991",
        "pre-commit>=3.0.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "dask>=2021.10.0",
        "fsspec>=2021.10.0",
        "pyarrow>=6.0.0",
    ],
    package_data={
        "": ["*.yaml", "*.json", "*.csv", "*.txt"],
    },
    include_package_data=True,
)
