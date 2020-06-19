# This is a Python template Makefile, do modification as you want
#
# Project:
# Author:
# Email :

HOST = 127.0.0.1
PYTHONPATH="$(shell printenv PYTHONPATH):$(PWD)"

## siamese experiments
run-1-siamese_exps:
	python ../src/training.py --quantization None --seed 1 --dataset "tiny-imagenet" --model "HorizontalNetworkV44" --results_path "/scratch/jlungu/siamese_exps" --data_path "/data/jlungu/siamese_cluster_data_new"
run-2-siamese_exps:
	python ../src/training.py --quantization "nullhop" --seed 1 --dataset "tiny-imagenet" --model "HorizontalNetworkV44" --results_path "/scratch/jlungu/siamese_exps" --data_path "/data/jlungu/siamese_cluster_data_new"
