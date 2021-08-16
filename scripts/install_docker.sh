#!/bin/bash

# activate environment
conda activate routed-fusion

# install all dependencies
cd deps

# install distance transform
cd distance-transform
CC=gcc pip install -e .
cd ..

# install graphics
cd graphics
CC=gcc pip install -e .
cd ..

# install tsdf
cd tsdf
CC=gcc pip install -e .
cd ..

