import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="thesis",
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    setup_requires = [],
    install_requires=[
        # "numpy",
        # "pandas",
        # # "scikit-fuzzy==0.4.2",
        # "dash==1.20.0",
        # # "matplotlib",
        # # "seaborn",
        # # "scikit-learn",
        # # "tensorflow==2.4.1",
        # "openpyxl",
        # # "torch",
        # # "torch-geometric",
        # # "torch-scatter",
        # # "torch-sparse",
        # # "captum",
        # "networkx"
    ],
    python_requires=">=3.6",
)
