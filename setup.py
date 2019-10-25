import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seqgra",
    version="0.0.1",
    author="Konstantin Krismer",
    author_email="krismer@mit.edu",
    license="MIT License",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kkrismer/seqgra",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={"": ["seqgra/data-config.xsd", "seqgra/model-config.xsd"]},
    install_requires=[
        "lxml>=4.4.1",
        "numpy>=1.14",
        "scikit-learn>=0.21.3",
        "matplotlib>=3.1.1",
        "scipy>=1.3.1",
        "pandas>=0.25.2"
    ],
    python_requires=">=3"
)
