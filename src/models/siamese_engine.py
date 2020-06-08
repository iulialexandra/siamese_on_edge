import os
import logging
import numpy.random as rng
import numpy as np
import time
from signal import SIGINT, SIGTERM
import tensorflow as tf
import tools.utils as util
import tools.visualization as vis
import data_processing.dataset_utils as dat
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from contextlib import redirect_stdout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from networks.horizontal_nets import *
from networks.original_nets import *
from networks.resnets import *
from data_processing.dataset_utils import tuple_to_dict

logger = logging.getLogger("siam_logger")


class SiameseEngine():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.evaluate_every = args.evaluate_every
        self.results_path = args.results_path
        self.model = args.model
        self.num_val_ways = args.n_val_ways
        self.val_trials = args.n_val_tasks
        self.image_dims = args.image_dims
        self.results_path = args.results_path
        self.plot_confusion = args.plot_confusion
        self.plot_training_images = args.plot_training_images
        self.plot_wrong_preds = args.plot_wrong_preds
        self.plot_val_images = args.plot_val_images
        self.plot_test_images = args.plot_test_images
        self.learning_rate = args.learning_rate
        self.lr_annealing = args.lr_annealing
        self.momentum_annealing = args.momentum_annealing
        self.momentum_slope = args.momentum_slope
        self.final_momentum = args.final_momentum
        self.optimizer = args.optimizer
        self.save_weights = args.save_weights
        self.checkpoint = args.chkpt
        self.left_classif_factor = args.left_classif_factor
        self.right_classif_factor = args.right_classif_factor
        self.siamese_factor = args.siamese_factor
        self.write_to_tensorboard = args.write_to_tensorboard
        self.summary_writer = tf.summary.create_file_writer(self.results_path)

    def setup_input(self, class_indices, num_samples, filenames, type):
        new_labels = list(range(len(class_indices)))
        if type == "train":
            dataset_chunk = self.batch_size
            shuffle = True

            initializer = tf.lookup.KeyValueTensorInitializer(
                tf.convert_to_tensor(class_indices, dtype=tf.int32),
                tf.convert_to_tensor(new_labels, dtype=tf.int32))

            self.table = tf.lookup.StaticHashTable(initializer, -1, name=type + "_table")

            self.table._initialize()

        else:
            shuffle = False
            if num_samples <= self.val_trials:
                dataset_chunk = num_samples
            else:
                dataset_chunk = min(num_samples, max(2 * len(class_indices), self.val_trials))



        dataset = dat.deploy_dataset(filenames,
                                     self.table,
                                     dataset_chunk,
                                     self.image_dims,
                                     shuffle)

        return dataset

    def setup_network(self, num_classes):
        if self.optimizer == 'sgd':
            optimizer = SGD(
                lr=self.learning_rate,
                momentum=0.5)
        elif self.optimizer == 'adam':
            optimizer = Adam(self.learning_rate)
        else:
            raise ("optimizer not known")

        model = util.str_to_class(self.model)
        siamese_network = model(self.image_dims, optimizer,
                                self.left_classif_factor,
                                self.right_classif_factor,
                                self.siamese_factor)
        self.net = siamese_network.build_net(num_classes)

        with open(os.path.join(self.results_path, 'modelsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.net.summary()

        if self.checkpoint:
            self.net.load_weights(self.checkpoint)

    def train(self, train_class_names, val_class_names, test_class_names, train_filenames,
              val_filenames, test_filenames, train_class_indices, val_class_indices,
              test_class_indices, num_val_samples, num_test_samples):

        num_train_cls = len(train_class_indices)
        self.setup_network(num_train_cls)

        # each sample is x, y
        # 

        train_dataset = self.setup_input(train_class_indices, None, train_filenames, 'train')

        val_dataset = self.setup_input(val_class_indices, num_val_samples, val_filenames, 'val')
        #
        # test_dataset = self.setup_input(test_class_indices, num_test_samples, test_filenames, 'test')

        self.net.fit(x=train_dataset, validation_data=val_dataset, epochs=self.num_epochs, verbose=1, shuffle=True)

    def validate(self, epoch, batch_index, train_metrics, left_images, right_images,
                 siamese_targets, val_inputs, val_targets, val_targets_one_hot, val_class_names,
                 test_inputs, test_targets, test_targets_one_hot, test_class_names):

        left_loss = np.mean(train_metrics["Left_branch_classification_loss"])
        left_acc = np.mean(train_metrics["Left_branch_classification_acc"])
        right_loss = np.mean(train_metrics["Right_branch_classification_loss"])
        right_acc = np.mean(train_metrics["Right_branch_classification_acc"])
        siamese_loss = np.mean(train_metrics["Siamese_classification_loss"])
        siamese_acc = np.mean(train_metrics["Siamese_classification_acc"])

        logger.info("Left branch classification loss and accuracy at the end of"
                    " epoch {}: {}, {}".format(epoch, left_loss, left_acc))
        logger.info("Right branch classification loss and accuracy at the end of"
                    " epoch {}: {}, {}".format(epoch, right_loss, right_acc))
        logger.info("Siamese classification loss and accuracy at the end of epoch"
                    " {}: {}, {}".format(epoch, siamese_loss, siamese_acc))

        epoch_folder = os.path.join(self.results_path, "epoch_{}".format(epoch))
        if not os.path.exists(epoch_folder):
            os.makedirs(epoch_folder)

        val_accuracy, val_y_pred, val_predictions, val_probs_std, val_probs_means, mean_delay, std_delay = self.eval(
            val_inputs,
            val_targets,
            val_class_names)
        test_accuracy, test_y_pred, test_predictions, test_probs_std, test_probs_means, mean_delay, std_delay = self.eval(
            test_inputs,
            test_targets,
            test_class_names)

        util.metrics_to_csv(os.path.join(epoch_folder, "metrics_epoch_{}.csv"
                                                       "".format(epoch)),
                            np.asarray([left_loss, left_acc, right_loss, right_acc,
                                        siamese_loss, siamese_acc, val_accuracy,
                                        test_accuracy]),
                            ["left_loss", "left_acc", "right_loss", "right_acc", "siamese_loss",
                             "siamese_acc", "siamese_val_accuracy", "siamese_test_accuracy"]
                            )

        if self.save_weights:
            # self.net.save_weights(os.path.join(self.results_path, "weights.h5"))
            self.net.save(os.path.join(self.results_path, "weights.h5"), overwrite=True, include_optimizer=False)

        if self.write_to_tensorboard:
            self._write_logs_to_tensorboard(batch_index, left_loss, left_acc, right_loss, right_acc,
                                            siamese_loss, siamese_acc, val_accuracy, test_accuracy,
                                            test_probs_std, test_probs_means)
        if epoch == 0:
            if self.plot_training_images:
                for b, batch_sample in enumerate(siamese_targets):
                    vis.plot_siamese_training_pairs(os.path.join(
                        self.results_path,
                        "siamese_training_sample_{}"
                        "".format(b)),
                        [left_images[b], right_images[b]], siamese_targets[b])

            if self.plot_val_images:
                for im in range(100):
                    vis.plot_validation_images(os.path.join(
                        self.results_path,
                        "siamese_validation_sample_{}.png"
                        "".format(im)),
                        [val_inputs[0][im], val_inputs[1][im]],
                        val_targets_one_hot[im])

            if self.plot_test_images:
                for im in range(100):
                    vis.plot_validation_images(os.path.join(
                        self.results_path,
                        "siamese_test_sample_{}.png"
                        "".format(im)),
                        [test_inputs[0][im], test_inputs[1][im]],
                        test_targets_one_hot[im])

        else:
            if self.plot_wrong_preds:
                val_incorrect_idx = np.where(val_predictions == 0)[0]
                for im in range(min(100, len(val_incorrect_idx))):
                    vis.plot_wrong_preds(os.path.join(epoch_folder,
                                                      "siamese_val_incorrect_sample_{}.png"
                                                      "".format(im)),
                                         [val_inputs[0][val_incorrect_idx[im]],
                                          val_inputs[1][val_incorrect_idx[im]]],
                                         val_targets_one_hot[val_incorrect_idx[im]],
                                         val_y_pred[val_incorrect_idx[im]])
                test_incorrect_idx = np.where(test_predictions == 0)[0]
                for im in range(min(100, len(test_incorrect_idx))):
                    vis.plot_wrong_preds(os.path.join(epoch_folder,
                                                      "siamese_test_incorrect_sample_{}.png"
                                                      "".format(im)),
                                         [test_inputs[0][test_incorrect_idx[im]],
                                          test_inputs[1][test_incorrect_idx[im]]],
                                         test_targets_one_hot[
                                             test_incorrect_idx[im]],
                                         test_y_pred[test_incorrect_idx[im]])

        if self.plot_confusion:
            cnf_matrix = confusion_matrix(val_targets, val_y_pred)
            vis.plot_confusion_matrix(epoch_folder, "val", cnf_matrix,
                                      classes=range(self.num_val_ways))
            cnf_matrix = confusion_matrix(test_targets, test_y_pred)
            vis.plot_confusion_matrix(epoch_folder, "test", cnf_matrix,
                                      classes=range(self.num_val_ways))

    def test(self, test_class_names, test_filenames, train_class_indices, test_class_indices, num_test_samples):
        num_train_cls = len(train_class_indices)
        self.setup_network(num_train_cls)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        test_inputs, test_targets, test_targets_one_hot = self.setup_input(sess, test_class_indices, num_test_samples,
                                                                           test_filenames, 'test')

        test_accuracy, test_y_pred, test_predictions, \
        test_probs_std, test_probs_means, delay, std_delay = self.eval(test_inputs, test_targets, test_class_names)

        np.savetxt(os.path.join(self.results_path, "inference.csv"), np.asarray([test_accuracy, delay, std_delay]),
                   ["test_acc", "mean_delay", "std_delay"])

    def eval(self, inps, targets, class_names):

        logger.info(
            "Evaluating model on {} random {} way one-shot learning tasks from classes {}"
            "...".format(self.val_trials, self.num_val_ways, class_names))

        y_pred = np.zeros((self.val_trials))
        probs_std = np.zeros((self.val_trials))
        probs_means = np.zeros((self.val_trials))
        timings = np.zeros((self.val_trials))

        for trial in range(self.val_trials):
            start = time.time()
            probs = self.net.predict([inps[0][trial], inps[1][trial]])[1]
            timings[trial] = time.time() - start
            y_pred[trial] = np.argmax(probs)
            probs_std[trial] = np.std(probs)
            probs_means[trial] = np.mean(probs)

        interm_acc = np.equal(y_pred, targets)
        tolerance = probs_std > 10e-8
        preds = np.logical_and(interm_acc, tolerance)
        accuracy = np.mean(preds)
        mean_delay = np.mean(timings[1:])
        std_delay = np.std(timings[1:])

        logger.info("{} way one-shot accuracy: {}% on classes {}"
                    "".format(self.num_val_ways, accuracy * 100, class_names))
        return accuracy, y_pred, preds, probs_std, probs_means, mean_delay, std_delay

    def _get_train_balanced_batch(self, images, labels, num_train_cls):
        with tf.device('/cpu:0'):
            labels = np.ravel(labels)
            targets = np.zeros((self.batch_size,), dtype=np.int8)
            shuffle_indices = rng.choice(range(self.batch_size), size=self.batch_size // 2)
            targets[shuffle_indices] = 1
            image_pairs = np.zeros((self.batch_size, 2,) + np.shape(images)[1:], dtype=np.float32)
            # choose the first row of images
            chosen_indices = rng.choice(range(len(labels)), size=self.batch_size, replace=False)
            left_labels = labels[chosen_indices]
            right_labels = []
            image_pairs[:, 0, :, :, :] = images[chosen_indices]
            for i, index in enumerate(chosen_indices):
                comparative_lbl = labels[index]
                if targets[i] == 0:
                    pair_index = rng.choice(np.where(labels != comparative_lbl)[0], size=(1,))[0]
                    image_pairs[i, 1, :, :, :] = images[pair_index]
                    right_labels.append(labels[pair_index])
                elif targets[i] == 1:
                    pair_index = rng.choice(np.where(labels == comparative_lbl)[0], size=(1,))[0]
                    image_pairs[i, 1, :, :, :] = images[pair_index]
                    right_labels.append(labels[pair_index])
            right_labels = np.array(right_labels)
            right_labels = to_categorical(right_labels, num_classes=num_train_cls)
            left_labels = to_categorical(left_labels, num_classes=num_train_cls)

            return (image_pairs[:, 0, :, :, :], image_pairs[:, 1, :, :, :],
                    left_labels, np.expand_dims(targets, 1), right_labels)

    def _make_oneshot_task(self, n_val_tasks, image_data, labels, n_ways):
        with tf.device('/cpu:0'):
            classes = np.unique(labels)
            assert len(classes) == n_ways
            if len(image_data) < n_val_tasks:
                replace = True
            else:
                replace = False
            reference_indices = rng.choice(range(len(labels)), size=(n_val_tasks,), replace=replace)
            reference_labels = np.ravel(labels[reference_indices])
            comparison_indices = np.zeros((n_val_tasks, n_ways), dtype=np.int32)
            targets = np.zeros((n_val_tasks, n_ways))
            targets[range(n_val_tasks), reference_labels] = 1
            for i, cls in enumerate(classes):
                cls_indices = np.where(labels == cls)[0]
                comparison_indices[:, i] = rng.choice(cls_indices, size=(n_val_tasks,),
                                                      replace=True)
            comparison_images = image_data[comparison_indices, :, :, :]
            reference_images = image_data[reference_indices, np.newaxis, :, :, :]
            reference_images = np.repeat(reference_images, n_ways, axis=1)
            image_pairs = [np.array(reference_images, dtype=np.float32),
                           np.array(comparison_images, dtype=np.float32)]
            return image_pairs, targets

    def _write_logs_to_tensorboard(self, batch_index, left_loss, left_acc, right_loss, right_acc,
                                   siamese_loss, siamese_acc, val_accuracy, test_accuracy,
                                   test_probs_std, test_probs_means):
        """ Writes the logs to a tensorflow log file
        """

        summary = tf.Summary()

        value = summary.value.add()
        value.simple_value = left_loss
        value.tag = 'Left images classification training loss'

        value = summary.value.add()
        value.simple_value = left_acc
        value.tag = 'Left images classification training accuracy'

        value = summary.value.add()
        value.simple_value = right_loss
        value.tag = 'Right images classification training loss'

        value = summary.value.add()
        value.simple_value = right_acc
        value.tag = 'Right images classification training accuracy'

        value = summary.value.add()
        value.simple_value = siamese_loss
        value.tag = 'Siamese training loss'

        value = summary.value.add()
        value.simple_value = siamese_acc
        value.tag = 'Siamese training accuracy'

        value = summary.value.add()
        value.simple_value = val_accuracy
        value.tag = 'Siamese validation accuracy'

        value = summary.value.add()
        value.simple_value = test_accuracy
        value.tag = 'Siamese testing accuracy on unseen classes'

        means_hist_summary = self._log_tensorboard_hist(test_probs_means,
                                                        'Siamese prediction probabilities means')
        std_hist_summary = self._log_tensorboard_hist(test_probs_std,
                                                      'Siamese prediction probabilities'
                                                      ' standard deviation')

        self.summary_writer.add_summary(summary, batch_index)
        self.summary_writer.add_summary(means_hist_summary, batch_index)
        self.summary_writer.add_summary(std_hist_summary, batch_index)
        self.summary_writer.flush()

    def _log_tensorboard_hist(self, values, tag, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        return summary
