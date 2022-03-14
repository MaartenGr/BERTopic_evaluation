from setuptools import setup, find_packages

test_packages = [
    "pytest>=5.4.3",
    "pytest-cov>=2.6.1"
]

docs_packages = [
    "mkdocs>=1.1",
    "mkdocs-material>=4.6.3",
    "mkdocstrings>=0.8.0",
]

base_packages = [
    "nltk>=3.2.4",
    "matplotlib>=3.4.3",
    "scikit_learn>=0.23.2",
    "srsly>=1.0.5",
    "octis==1.10.2"
]

bertopic_packages = [
    "bertopic==0.9.4"

]

top2vec_packages = [
    "top2vec==1.0.26"
]

ctm_packages = [
    "contextualized_topic_models==2.2.1"
]


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="evaluation",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="0.1.0",
    author="Maarten P. Grootendorst",
    author_email="maartengrootendorst@gmail.com",
    description="Evaluation of topic modeling techniques.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaartenGr/Evaluation",
    keywords="nlp bert topic modeling embeddings",
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=base_packages,
    extras_require={
        "test": test_packages,
        "docs": docs_packages,
        "bertopic": bertopic_packages,
        "top2vec": top2vec_packages,
        "ctm": ctm_packages,
    },
    python_requires='>=3.7',
)
