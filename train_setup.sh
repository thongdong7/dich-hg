#!/usr/bin/env bash

# pip install datasets evaluate sacrebleu
# pip install -U accelerate
# pip install -U transformers


pip uninstall -y -q tensorflow keras tensorflow-estimator tensorflow-text tf-keras
pip install protobuf~=3.20.3
pip install -q "tensorflow-text==2.11.0"
# pip install -q tensorflow_datasets
pip install -q "tensorflow==2.11.1" tensorflow_datasets matplotlib
pip list|grep tensorflow