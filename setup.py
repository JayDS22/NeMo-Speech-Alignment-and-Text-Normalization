from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="nemo-speech-alignment",
    version="1.0.0",
    author="Jay Guwalani",
    description="Speech Alignment & Text Normalization using NVIDIA NeMo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "num2words>=0.5.13",
        "jiwer>=3.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "full": [
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "streamlit>=1.28.0",
            "plotly>=5.18.0",
            "matplotlib>=3.7.0",
            "pandas>=2.0.0",
        ],
        "nemo": [
            "nemo_toolkit[asr]>=1.22.0",
            "torch>=2.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nemo-align=src.pipeline:main",
        ],
    },
)
