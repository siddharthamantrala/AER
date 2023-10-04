# SHARC

Code for preprint paper "Saliency-Guided Hidden Associative Replay for Continual Learning"

[sharc_overall.pdf](https://github.com/AlexQilong/SHARC/files/12799712/sharc_overall.pdf)

We extended the [Mammoth](https://github.com/aimagelab/mammoth) framework with our method SHARC

## Setup

[Anaconda](https://www.anaconda.com/download) is recommended for virtual environment management.

After installation, do:
+ `conda create -n SHARC python=3.10.0`
+ `conda activate SHARC`
+ `pip install -r requirements.txt`

## Run

Example:

`python main.py --model 'er_bpcn' --buffer_size 200 --dataset 'seq-cifar100' --theta 0.1 --seed 0`
