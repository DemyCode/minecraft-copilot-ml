[![Main pipeline](https://github.com/DemyCode/minecraft-copilot-ml/actions/workflows/master.yml/badge.svg?branch=master)](https://github.com/DemyCode/minecraft-copilot-ml/actions/workflows/master.yml)

https://demycode.github.io/minecraft-copilot-ml/master/

# Minecraft Copilot ML

This repository is the Machine Learning / Data Science part of the minecraft-copilot mod.
The goal of this repository is to research the optimal Machine Learning model for the minecraft copilot.

### Usefull Links :

- [MLFlow Registry](google.com): Register your model metrics and artifacts on this registry
- [Public Data](http://minecraft-schematics-raw.s3.amazonaws.com/): Raw data used to train the model (schematics files of structures)
- [Public Data (Processed 16x16x16)](http://minecraft-schematics-16.s3.amazonaws.com/): 16x16x16 cubes of blocks created from the raw data
- [Docker Image Link](https://gallery.ecr.aws/p3u9i4c1/minecraft-copilot-ml): Registry of images

## Requirements

- [Python 3.10](https://www.python.org/downloads/release/python-3100/)
- [pip](https://pypi.org/project/pip/)
- [docker](https://docs.docker.com/desktop/)

## Local Installation

NOTE : We highly encourage you to use a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/#how-can-you-work-with-a-python-virtual-environment).

```bash
pip install poetry
poetry export -f requirements.txt --output requirements.txt --without-hashes
pip install -r requirements.txt
rm requirements.txt
```


## Docker usage

```
docker build . -t minecraft-copilot-ml:latest
docker run minecraft-copilot-ml:latest
```

## Development

```bash
pip install poetry
poetry export --dev -f requirements.txt --output requirements.txt --without-hashes
pip install -r requirements.txt
rm requirements.txt
```
