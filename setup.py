import setuptools
import seqgra

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seqgra",
    version=seqgra.__version__,
    author="Konstantin Krismer",
    author_email="krismer@mit.edu",
    license="MIT License",
    description="Synthetic rule-based biological sequence data generation for architecture evaluation and search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kkrismer/seqgra",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3"
    ],
    package_data={seqgra.__name__: ["data-config.xsd", "model-config.xsd",
                                    "evaluator/plotagreement.R",
                                    "simulator/plotgrammarpositionheatmap.R"]},
    install_requires=[
        "Cython>=0.29",
        "lxml>=4.4.1",
        "matplotlib>=3.1",
        "numpy>=1.14",
        "pandas>=0.25",
        "PyYAML>=5.3",
        "scikit-image>=0.16",
        "scikit-learn>=0.21",
        "scipy>=1.3",
        "setuptools>=41.6",
        "ushuffle>=1.1.2"
    ],
    python_requires=">=3",
    entry_points={
        "console_scripts": ["seqgra=seqgra.seqgra:main",
                            "metaseqgra=seqgra.metaseqgra:main"]
    }
)
