#!/bin/sh

rm -rf gag_python/models
mkdir -p gag_python/models  && \
cd gag_python/models  && \
wget https://github.com/devProgst/gag-python/raw/main/gag-models.zip && \
unzip gag-models.zip  && \
rm gag-models.zip
