import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="foamliu", # Replace with your own username
    version="0.0.1",
    author="Yang Liu",
    author_email="foamliu@yeah.net",
    description="Free Face SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/foamliu/FaceSDK",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)