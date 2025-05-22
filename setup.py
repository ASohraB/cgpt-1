from setuptools import setup, find_packages

setup(
    name="cgpt-1",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "numpy",
    ],
    author="Your Name",
    description="Normalized Complex-valued GPT-package",
    python_requires=">=3.8",
)