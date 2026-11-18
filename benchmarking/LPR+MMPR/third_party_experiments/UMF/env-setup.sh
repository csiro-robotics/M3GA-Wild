#!/bin/bash
# Source - https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/third_party/how":$PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/third_party/cnnimageretrieval-pytorch-1.2":$PYTHONPATH