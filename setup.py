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
    package_data={"": ["seqgra/config.xsd"]},
    install_requires=[
        "lxml>=4.4.1",
        "numpy>=1.14"
    ],
    python_requires=">=3"
)
