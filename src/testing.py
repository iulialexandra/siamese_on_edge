import argparse
import os
import json
import time
import numpy.random as rng
import tools.utils as util
import data_processing.dataset_utils as dat
import numpy as np
from models.siamese_engine import SiameseEngine
import tensorflow as tf
import sys

def main(args):
    args.write_to_tensorboard = False
    args.save_weights = True
    args.console_print = True
    args.num_epochs = 100
    args.n_val_ways = 5
    args.evaluate_every = 10
    args.n_val_tasks = 1000
    args.batch_size = 32

    args.final_momentum = 0.9
    args.momentum_slope = 0.01
    args.learning_rate = 0.001
    args.lr_annealing = True
    args.momentum_annealing = True
    args.optimizer = "sgd"

    args.left_classif_factor = 0.7
    args.right_classif_factor = 0.7
    args.siamese_factor = 1.
    args.dataset = "tiny-imagenet"
    args.model = "HorizontalNetworkV5"
    args.data_path = "/mnt/data/siamese_cluster_new/data"

    if args.dataset == "mnist":
        args.image_dims = (28, 28, 1)
    elif args.dataset == "omniglot":
        args.image_dims = (105, 105, 1)
    elif args.dataset == "cifar100":
        args.image_dims = (32, 32, 3)
    elif args.dataset == "roshambo":
        args.image_dims = (64, 64, 1)
    elif args.dataset == "tiny-imagenet":
        args.image_dims = (64, 64, 3)
    elif args.dataset == "mini-imagenet":
        args.image_dims = (84, 84, 3)
    else:
        print(" Dataset not supported.")
    args.dataset_path = os.path.join(args.data_path, args.dataset)

    args, logger = util.initialize_experiment(args)
    siamese = SiameseEngine(args)
    (train_class_names, val_class_names, test_class_names, train_filenames,
    val_filenames, test_filenames, train_class_indices, val_class_indices,
    test_class_indices, num_val_samples, num_test_samples) = dat.read_dataset_csv(args.dataset_path, args.n_val_ways)
    siamese.test(test_class_names, test_filenames, train_class_indices, test_class_indices, num_test_samples)

def parse_args():
    """Parses arguments specified on the command-line
    """
    argparser = argparse.ArgumentParser('Train and eval  siamese networks')

    argparser.add_argument('--batch_size', type=int,
                           help="The number of images to process at the same time",
                           default=32)
    argparser.add_argument('--n_val_tasks', type=int,
                           help="how many one-shot tasks to validate on",
                           default=1000)
    argparser.add_argument('--n_val_ways', type=int,
                           help="how many support images we have for each image to be classified",
                           default=5)
    argparser.add_argument('--num_epochs', type=int,
                           help="Number of training epochs",
                           default=200)
    argparser.add_argument('--evaluate_every', type=int,
                           help="interval for evaluating on one-shot tasks",
                           default=5)
    argparser.add_argument('--momentum_slope', type=float,
                           help="linear epoch slope evolution",
                           default=0.01)
    argparser.add_argument('--final_momentum', type=float,
                           help="Final layer-wise momentum (mu_j in the paper)",
                           default=0.9)
    argparser.add_argument('--learning_rate', type=float,
                           default=0.001)
    argparser.add_argument('--seed', help="Random seed to make experiments reproducible",
                           type=int, default=13)
    argparser.add_argument('--left_classif_factor', help="How much left classification loss is"
                                                         " weighted in the total loss",
                           type=float, default=0.7)
    argparser.add_argument('--right_classif_factor', help="How much right classification loss is"
                                                          " weighted in the total loss",
                           type=float, default=0.7)
    argparser.add_argument('--siamese_factor', help="How much the siamese similarity should count",
                           type=float, default=1.)
    argparser.add_argument('--lr_annealing',
                           help="If set to true, it changes the learning rate at each epoch",
                           type=bool, default=True)
    argparser.add_argument('--momentum_annealing',
                           help="If set to true, it changes the momentum at each epoch",
                           type=bool, default=True)
    argparser.add_argument('--optimizer',
                           help="The optimizer to use for training",
                           type=str, default='sgd')
    argparser.add_argument('--console_print',
                           help="If set to true, it prints logger info to console.",
                           type=bool, default=False)
    argparser.add_argument('--plot_training_images',
                           help="If set to true, it plots input training data",
                           type=bool, default=False)
    argparser.add_argument('--plot_val_images',
                           help="If set to true, it plots input validation data",
                           type=bool, default=False)
    argparser.add_argument('--plot_test_images',
                           help="If set to true, it plots input test data",
                           type=bool, default=False)
    argparser.add_argument('--plot_confusion',
                           help="If set to true, it plots the confusion matrix",
                           type=bool, default=False)
    argparser.add_argument('--plot_wrong_preds',
                           help="If set to true, it plots the images that were predicted wrongly",
                           type=bool, default=False)
    argparser.add_argument('--results_path',
                           help="Path to results. If none, the folder gets created with"
                                "current date and time", default=None)
    argparser.add_argument('--chkpt',
                           help="Path where the weights to load are",
                           default=None)
    argparser.add_argument('--model', type=str, default="OriginalNetworkV2")
    argparser.add_argument('--save_weights',
                           help="Whether to save the weights at every evaluation",
                           type=bool, default=True)
    argparser.add_argument('--write_to_tensorboard',
                           help="Whether to save the results in a tensorboard-readable format",
                           type=bool, default=False)
    argparser.add_argument('--dataset',
                           help="The dataset of choice", type=str,
                           default="roshambo")
    argparser.add_argument('--data_path',
                           help="Path to data", type=str,
                           default="/mnt/data/siamese_cluster_new/data")
    return argparser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "mnist":
        args.image_dims = (28, 28, 1)
    elif args.dataset == "omniglot":
        args.image_dims = (105, 105, 1)
    elif args.dataset == "cifar100":
        args.image_dims = (32, 32, 3)
    elif args.dataset == "roshambo":
        args.image_dims = (64, 64, 1)
    elif args.dataset == "tiny-imagenet":
        args.image_dims = (64, 64, 3)
    elif args.dataset == "mini-imagenet":
        args.image_dims = (84, 84, 3)
    else:
        print(" Dataset not supported.")

    args.dataset_path = os.path.join(args.data_path, args.dataset)
    main(args)
