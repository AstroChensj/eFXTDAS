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
    packages=find_packages(
        include=[
            "fxtcaldb",
            "fxtcaldb.*",
            "fxtpsfgen",
            "fxtpsfgen.*",
            "fxtsrcdet",
            "fxtsrcdet.*",
            "fxtregions",
            "fxtregions.*",
            "fxtrspgen",
            "fxtrspgen.*",
            "fxtcombine",
            "fxtcombine.*",
            "fxtbkgoptrate",
            "fxtbkgoptrate.*",
        ]
    ),
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "reproject",
        "regions",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "fxtsrcdet=fxtsrcdet.pipeline:main",
            "fxtregions=fxtregions.pipeline:main",
            "fxtrspgen=fxtrspgen.pipeline:main",
            "fxtpsfgen=fxtpsfgen.pipeline:main",
            "fxtcombine=fxtcombine.pipeline:main",
            "fxtcombine-quickview=fxtcombine.quickview:main",
            "fxtbkgoptrate=fxtbkgoptrate.pipeline:main",
        ]
    },
)
