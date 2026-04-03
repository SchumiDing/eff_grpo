#!/bin/bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
# install torch
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# install FA2 and diffusers
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation 

pip install -r requirements-lint.txt

# install fastvideo
pip install -e .

pip install ml-collections absl-py inflect==6.0.4 pydantic==1.10.9 huggingface_hub==0.24.0 protobuf==3.20.0 
