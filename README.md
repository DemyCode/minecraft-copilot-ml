[![Main pipeline](https://github.com/DemyCode/minecraft-copilot-ml/actions/workflows/master.yml/badge.svg?branch=master)](https://github.com/DemyCode/minecraft-copilot-ml/actions/workflows/master.yml)

https://demycode.github.io/minecraft-copilot-ml/master/

# Minecraft Copilot ML

This repository is the Machine Learning / Data Science part of the minecraft-copilot mod.
The goal of this repository is to research the optimal Machine Learning model for the minecraft copilot.

### Usefull Links :

- [Public Data](http://minecraft-schematics-raw.s3.amazonaws.com/): Raw data used to train the model (schematics files of structures)
- [Docker Image Link](https://gallery.ecr.aws/p3u9i4c1/minecraft-copilot-ml): Registry of images

## Requirements

- [Python 3.10](https://www.python.org/downloads/release/python-3100/)
- [pip](https://pypi.org/project/pip/)

If running from Docker :
- [docker](https://docs.docker.com/desktop/)
- [NVIDIA Graphics Card](https://en.wikipedia.org/wiki/CUDA)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#prerequisites)

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
docker build . -t minecraft-copilot-ml
docker run minecraft-copilot-ml
```

## Development

```bash
pip install poetry
poetry export --dev -f requirements.txt --output requirements.txt --without-hashes
pip install -r requirements.txt
rm requirements.txt
```
