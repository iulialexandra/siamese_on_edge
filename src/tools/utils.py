import os
import sys
import logging
import json
import datetime
import csv
import numpy as np
import signal
from random import shuffle
from pathlib import Path
from networks.original_nets import *
from networks.horizontal_nets import *
import random as rn

import time
import logging

logger = logging.getLogger()


class Uninterrupt(object):
    """From https://gist.github.com/nonZero/2907502
    Use as:
    with Uninterrupt() as u:
        while not u.interrupted:
            # train
    """
    def __init__(self, sigs=[signal.SIGINT], verbose=False):
        self.sigs = sigs
        self.verbose = verbose
        self.interrupted = False
        self.orig_handlers = None

    def __enter__(self):
        if self.orig_handlers is not None:
            raise ValueError("Can only enter `Uninterrupt` once!")

        self.interrupted = False
        self.orig_handlers = [signal.getsignal(sig) for sig in self.sigs]

        def handler(signum, frame):
            self.release()
            self.interrupted = True
            if self.verbose:
                print("Interruption scheduled...", flush=True)

        for sig in self.sigs:
            signal.signal(sig, handler)

        return self

    def __exit__(self, type_, value, tb):
        self.release()

    def release(self):
        if self.orig_handlers is not None:
            for sig, orig in zip(self.sigs, self.orig_handlers):
                signal.signal(sig, orig)
            self.orig_handlers = None


def str_to_class(str):
    """Gets a class when given its name.

    Args:
        str: the name of the class to retrieve
    """
    return getattr(sys.modules[__name__], str)


def make_results_dir(args):
    """Makes one folder for the results using the current date and time, another that holds
    the most recent experiment data and initializes the logger.
    """
    now = datetime.datetime.now()
    date = "{}_{}_{}-{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                         now.second, now.microsecond)
    if args.results_path is None:
        parent_path = Path().absolute().parent
        save_path = os.path.join(parent_path, "results")
    else:
        save_path = args.results_path
    if args.left_classif_factor > 0:
        args.with_classif = "yes"
    else:
        args.with_classif = "no"
    results_path = os.path.join(save_path, str(date) + "_seed_" + str(args.seed) + "_"
                                + args.dataset + "_" + args.model + "_" + args.with_classif)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    logger = initialize_logger(results_path, args.console_print)
    return results_path, logger


def experiment_details(args):
    """Writes the arguments to file to keep track of various experiments"""
    exp_path = os.path.join(args.results_path, "experiment_details.json")
    if not os.path.exists(exp_path):
        with open(exp_path, "w") as outfile:
            json.dump(vars(args), outfile)


def initialize_logger(output_dir, print_to_console):
    """initializes loggers for debug and error, prints to file

    Args:
        output_dir: the directory of the logger file
    """

    logger = logging.getLogger("siam_logger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = 0
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                  "%d-%m-%Y %H:%M:%S")
    # Setup file logging
    fh = logging.FileHandler(os.path.join(output_dir, "log.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Setup console logging
    if print_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def initialize_experiment(args, train=True):
    if train:
        args.results_path, logger = make_results_dir(args)
        experiment_details(args)
    else:
        args.results_path = args.checkpoint
        logger = initialize_logger(args.results_path, True)

    # make the experiment deterministic
    np.random.seed(args.seed)
    rn.seed(args.seed)
    tf.set_random_seed(args.seed)
    os.environ["PYTHONSEED"] = str(args.seed)
    return args, logger


def sweep_directories(directory_list, token=None):
    """Given a list of directories, it returns the files inside"""
    files = []
    if type(directory_list) != list:
        directory_list = [os.path.join(directory_list, direc) for direc in
                          os.listdir(directory_list)]
    for directory in directory_list:
        for f in os.listdir(directory):
            file = os.path.join(directory, f)
            if (os.path.isfile(file) and token in file):
                files.append(file)
    shuffle(files)
    return files


def timed(method):
    def timeit(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        elapsed_time = end_time - start_time
        logger.info("{} run completed in {:.3f}s".format(method.__name__, elapsed_time))
        return result

    return timeit


def metrics_to_csv(filepath, values, fieldnames):
    with open(filepath, 'w') as csvfile:
        values_dict = dict(zip(fieldnames, values))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(values_dict)


def nearest_neighbour_correct(pairs, targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt(pairs[0][i] ** 2 - pairs[1][i] ** 2))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0
