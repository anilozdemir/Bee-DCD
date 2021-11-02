import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BTDM",
    version="1.0",
    author="Anil Ozdemir",
    author_email="a.ozdemir@sheffield.ac.uk",
    description="Collection of source-code for Bee Temporal Decision Making study",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anilozdemir/Bee-DCD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
