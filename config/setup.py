# setup.py
from setuptools import setup, find_packages
import os

# Read README
with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

setup(
    name="hybrid-fraud-detection",
    version="1.0.0",
    author="Hybrid Fraud Detection Team",
    author_email="contact@hybridfraud.ai",
    description="Hybrid Credit Card Fraud Detection System combining Autoencoder with ELECTRA-inspired Active Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/hybrid-fraud-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "coverage>=6.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "bokeh>=2.4.0",
        ],
        "experiments": [
            "optuna>=3.0.0",
            "mlflow>=1.20.0",
            "wandb>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "hybrid-fraud-detection=fraud_detection_system:main",
            "fraud-quick-experiment=fraud_detection_system:quick_fraud_detection_experiment",
        ],
    },
    include_package_data=True,
    package_data={
        "hybrid_fraud_detection": [
            "config/*.yaml",
            "data/sample/*.csv",
            "notebooks/*.ipynb",
        ],
    },
)