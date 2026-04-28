from setuptools import setup, find_packages

setup(
    name="helix-trefoil-loss",
    version="1.0.0",
    description="A PyTorch topological regularizer based on the Helix-TTD Constitutional Hamiltonian.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Stephen Hope & The Helix Commonwealth",
    author_email="steve@helixprojectai.com",
    url="https://github.com/helixprojectai-code/helix-trefoil-loss",
    packages=find_packages(),
    install_requires=["torch>=1.9.0", "numpy>=1.20.0"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
)
