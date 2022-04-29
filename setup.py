from setuptools import setup, find_packages


with open("npf/__init__.py") as init_file:
    __version__ = ""
    # extract __version__
    for line in init_file:
        if line.startswith("__version__"):
            exec(line)
            break


with open("README.md") as readme_file:
    readme = readme_file.read()


setup(
    name="np-family",
    version=__version__,
    author="EungGu Yun",
    author_email="yuneg11@gmail.com",
    description="Neural Processes implementations in JAX and PyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/yuneg11/Neural-Process-Family",
    project_urls={
        "Documentation": "https://yuneg11.github.io/Neural-Process-Family",
        "Source": "https://github.com/yuneg11/Neural-Process-Family",
        "Tracker": "https://github.com/yuneg11/Neural-Process-Family/issues"
    },
    packages=find_packages(include=["npf", "npf.*"]),
    package_dir={"npf": "npf"},
    # package_data={'': []},
    # include_package_data=True,
    license="MIT license",
    python_requires=">=3.7",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
