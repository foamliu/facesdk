import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="facesdk",  # Replace with your own username
    version="0.0.4",
    author="Yang Liu",
    author_email="foamliu@yeah.net",
    url="https://github.com/foamliu/FaceSDK",
    packages=setuptools.find_packages(),
    package_data={
        'facesdk': ['model.pt'],
        '': ['mobilenet0.25_Final.pth'],
    },
    description="Face SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms=["all"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords='face detection recognition',
    python_requires='>=3.5',
    install_requires=[
        'torch >= 1.0.0',
        'torchvision',
        'opencv-python'
    ],
)
