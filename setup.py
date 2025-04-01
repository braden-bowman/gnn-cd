from setuptools import setup, find_packages

setup(
    name="community_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "networkx>=2.8.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "rustworkx>=0.13.0", 
        "polars>=0.19.3",
        "pyarrow>=14.0.1",
    ],
    extras_require={
        "gnn": [
            "torch>=1.10.0",
            "torch-geometric>=2.0.0",
            "torch-sparse>=0.6.16",
            "torch-scatter>=2.1.0",
        ],
        "traditional": [
            "python-louvain>=0.16",
            "cdlib>=0.2.0",
        ],
        "visualization": [
            "plotly>=5.5.0",
        ],
        "gui": [
            "nicegui>=1.3.0",
        ],
        "all": [
            "torch>=1.10.0",
            "torch-geometric>=2.0.0",
            "torch-sparse>=0.6.16",
            "torch-scatter>=2.1.0",
            "python-louvain>=0.16",
            "cdlib>=0.2.0",
            "plotly>=5.5.0",
            "rustworkx>=0.13.0",
            "polars>=0.19.3",
            "pyarrow>=14.0.1",
            "nicegui>=1.3.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'gnn-cd-gui=community_detection.gui:run_gui',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive framework for community detection in graphs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/community-detection-framework",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
)
