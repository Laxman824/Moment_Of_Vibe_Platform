from setuptools import setup, find_packages

setup(
    name="emotion-analytics",
    version="0.1.0",
    description="Emotion Analytics Module for Moment of Vibe MVP",
    author="AI/ML Engineer",
    python_requires=">=3.12",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.2",
        "pandas>=2.1.4",
        "scikit-learn>=1.3.2",
        "opensmile>=2.6.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "torch>=2.5.0",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "pytest>=7.4.3",
        "pytest-cov>=4.1.0",
        "tqdm>=4.66.1",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": ["jupyter", "ipykernel", "black", "flake8", "mypy"],
    },
)