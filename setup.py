from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="torchrec-torchgeometric",
    version="0.1.0",
    author="Ethan Henley",
    author_email="ethan.henley@techolution.com",
    description="A cookbook for building recommendation systems with TorchRec and PyTorch Geometric",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ethanshenley/torchrec-pytorch-geometric-cookbook",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)