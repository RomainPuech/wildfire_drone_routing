from setuptools import setup, find_packages

setup(
    name="wildfire_drone",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-python",
        "Pillow",
        "torch",
        "tqdm",
        "scikit-learn",
        "scipy",
        "juliacall",
    ],
    entry_points={
        "console_scripts": [
            "wildfire-eval=wildfire_drone.benchmark:run_benchmark_for_strategy"
        ]
    },
    author="Joseph Ye",
    description="A wildfire detection drone routing library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RomainPuech/wildfire_drone_routing",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
