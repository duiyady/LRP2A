# LRP2A: Layer-wise Relevance Propagation based Adversarial Attacking for Graph Neural Networks

This implementation is written in Python 3 and uses Tensorflow2
## Requirements
* 'python=3.7.7'
* 'matplotlib'
* 'numpy=1.18.5'
* 'tensorflow=2.2.0'

## Data processing
First, you need to run data_conversion.py in the corresponding data folder, such as ./graph_data/cora/data_conversion.py

## Run the code
You can run 1_example_attack.py, modify the variables of the dataset and model_name at the top of this file to perform the corresponding attacks.

To facilitate performance verification, we provide simple verification code. modify the variables of the dataset, model_name, DIRECT, PER_TYPE and PER_NUM at the top of these file to perform the corresponding attacks. where PER_TYPE and PER_NUM control budget.
* PER_TYPE = 0, budget = PER_NUM
* PER_TYPE = 1, budget = degree + PER_NUM
* PER_TYPE = 2, budget = degree * PER_NUM

## Email
duiyady@163.com
