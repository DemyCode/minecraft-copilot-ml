[![Main pipeline](https://github.com/DemyCode/minecraft-copilot-ml/actions/workflows/master.yml/badge.svg?branch=master)](https://github.com/DemyCode/minecraft-copilot-ml/actions/workflows/master.yml)

# Minecraft Copilot ML

This repository is the Machine Learning / Data Science part of the minecraft-copilot mod.
The goal of this repository is to research the optimal Machine Learning model for the minecraft copilot.
Once found the main pipeline will train the model and register the artifact in MLFlow registry for later use.
### Usefull Links :

- [MLFlow Registry](google.com): Register your model metrics and artifacts on this registry
- [Public Data](http://minecraft-schematics-raw.s3.amazonaws.com/): Raw data used to train the model (schematics files of structures)
- [Public Data (Processed 16x16x16)](http://minecraft-schematics-16.s3.amazonaws.com/): 16x16x16 cubes of blocks created from the raw data

## Requirements

- [Python 3.10](google.com)
- [pip](google.com)
- [docker](google.com)

## Local Installation

```bash
pip install poetry
poetry export -f requirements.txt --output requirements.txt --without-hashes
pip install -r requirements.txt
rm requirements.txt
```

NOTE : We highly encourage you to use a [virtual environment](https://google.com).

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
