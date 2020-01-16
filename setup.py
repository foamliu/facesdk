import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="facesdk",  # Replace with your own username
    version="0.0.1",
    author="Yang Liu",
    author_email="foamliu@yeah.net",
    description="Face SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/foamliu/FaceSDK",
    packages=setuptools.find_packages(),
    platforms=["all"],
    install_requires=['torch', 'torchvision', 'opencv-python'],
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
    python_requires='>=3.5',
    package_data={
        'facesdk': ['model.pt'],
        'retinaface': ['weights/mobilenet0.25_Final.pth'],
    },
)
