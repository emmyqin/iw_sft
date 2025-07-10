# Importance Weighted Supervised Fine Tuning (iw-SFT)

Author's Pytorch implementation of **I**mportance **W**eighted **S**upervised **F**ine **T**uning (iw-SFT). Iw-SFT uses importance weights to adaptively upweight or downweight points during training; we show this provides a much tighter bound to the RL training objective in comparison to SFT alone.


## Overview of the Code
There are two python files, bounding_trainers.py contains iw-sft interpreted and described in our paper. In order to run this code first you need to install requirements.txt. We recommend using uv such as below:
~~~
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
~~~

### Running the code
To run this code simply do:
~~~
./iw_sft.sh
~~~
Make sure to set up your wandb when you first run.

### Evaluating the code
This code is cloned from the [S1](https://github.com/simplescaling/s1) repository to run:
~~~
cd eval/lm-evaluation-harness
pip install -e .[math,vllm]
~~~
