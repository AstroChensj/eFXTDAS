from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"


setup(
    name="eFXTDAS",
    version="0.1.0",
    description="Enhanced data analysis tools for Einstein Probe FXT, extending the official FXTDAS workflow.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Shi-Jiang Chen",
    author_email="JohnnyCsj666@gmail.com",
    url="https://github.com/AstroChensj/eFXTDAS.git",
    python_requires=">=3.10",
    packages=find_packages(
        include=[
            "fxtsrcdet",
            "fxtsrcdet.*",
            "fxtregions",
            "fxtregions.*",
            "fxteefmap",
            "fxteefmap.*",
            "fxtcombine",
            "fxtcombine.*",
            "fxtpsf_helpers",
            "fxtpsf_helpers.*",
        ]
    ),
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "reproject",
        "regions",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "fxtsrcdet=fxtsrcdet.pipeline:main",
            "fxtregions=fxtregions.pipeline:main",
            "fxteefmap=fxteefmap.pipeline:main",
            "fxtcombine=fxtcombine.pipeline:main",
        ]
    },
)
