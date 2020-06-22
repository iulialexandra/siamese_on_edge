import os
import logging
import numpy.random as rng
import numpy as np
import time
from signal import SIGINT, SIGTERM
import tensorflow as tf
import tools.utils as util
import data_processing.dataset_utils as dat
from contextlib import redirect_stdout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from networks.horizontal_nets import *
from tensorflow.keras.callbacks import ReduceLROnPlateau

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
        self.quantization = args.quantization
        self.write_to_tensorboard = args.write_to_tensorboard
        self.logs_dir = os.path.join(self.results_path, "logs")
        self.summary_writer = tf.summary.create_file_writer(os.path.join(self.logs_dir, "validate"))

    def setup_input_train(self, class_indices, filenames):
        new_labels = list(range(len(class_indices)))
        initializer = tf.lookup.KeyValueTensorInitializer(
            tf.convert_to_tensor(class_indices, dtype=tf.int32),
            tf.convert_to_tensor(new_labels, dtype=tf.int32))
        self.table = tf.lookup.StaticHashTable(initializer, -1, name="train_table")
        self.table._initialize()
        dataset = dat.deploy_dataset_train(filenames,
                                           self.table,
                                           self.batch_size,
                                           self.image_dims,
                                           shuffle=True)
        return dataset

    def setup_input_test(self, class_indices, num_samples, filenames, type):
        new_labels = list(range(len(class_indices)))
        initializer = tf.lookup.KeyValueTensorInitializer(
            tf.convert_to_tensor(class_indices, dtype=tf.int32),
            tf.convert_to_tensor(new_labels, dtype=tf.int32))
        self.table = tf.lookup.StaticHashTable(initializer, -1, name=type + "_table")
        self.table._initialize()

        if num_samples <= self.val_trials:
            dataset_chunk = num_samples
        else:
            dataset_chunk = min(num_samples, max(2 * len(class_indices), self.val_trials))

        dataset = dat.get_dataset_from_filename(filenames, self.table, self.image_dims, shuffle=True)
        dataset.batch(dataset_chunk)
        list_dataset = np.array(list(dataset.as_numpy_iterator()))
        images = np.array([it for it in list_dataset[:, 0]])
        labels = np.array([it for it in list_dataset[:, 1]])
        image_pairs, targets, targets_one_hot = self._make_oneshot_task(self.val_trials, images, labels,
                                                                        self.num_val_ways)
        return image_pairs, targets, targets_one_hot

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
        siamese_network = model(self.image_dims)
        self.net_train, self.net_test = siamese_network.build_net(num_classes, self.quantization)

        self.net_train.compile(loss={"Left_branch_classification": "categorical_crossentropy",
                                  "Siamese_classification": "binary_crossentropy",
                                  "Right_branch_classification": "categorical_crossentropy"},
                            optimizer=optimizer,
                            metrics={"Left_branch_classification": "accuracy",
                                     "Siamese_classification": "accuracy",
                                     "Right_branch_classification": "accuracy"},
                            loss_weights={"Left_branch_classification": self.left_classif_factor,
                                          "Siamese_classification": self.siamese_factor,
                                          "Right_branch_classification": self.right_classif_factor})

        with open(os.path.join(self.results_path, 'modelsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.net_train.summary()

        if self.checkpoint:
            self.net_train.load_weights(os.path.join(self.checkpoint, "saved_model_train/weights.h5"))


    def train(self, train_class_names, val_class_names, test_class_names, train_filenames,
              val_filenames, test_filenames, train_class_indices, val_class_indices,
              test_class_indices, num_val_samples, num_test_samples):
        num_train_cls = len(train_class_indices)
        self.setup_network(num_train_cls)
        train_dataset = self.setup_input_train(train_class_indices, train_filenames)
        val_inputs, val_targets, val_targets_one_hot = self.setup_input_test(val_class_indices, num_val_samples,
                                                                             val_filenames, 'val')
        test_inputs, test_targets, test_targets_one_hot = self.setup_input_test(test_class_indices, num_test_samples,
                                                                                test_filenames, 'test')

        lr_reducer = ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            cooldown=1,
            patience=5,
            min_lr=0.5e-6)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logs_dir,
                                                              histogram_freq=0,
                                                              write_graph=True,
                                                              write_images=False, update_freq='batch',
                                                              profile_batch=0, embeddings_freq=0,
                                                              embeddings_metadata=None )

        callbacks = [lr_reducer, tensorboard_callback]

        for epoch in range(0, self.num_epochs, self.evaluate_every):
            logger.info("Training epoch {} ...".format(epoch))

            self.validate(epoch, val_inputs, val_targets, val_targets_one_hot, val_class_names, test_inputs,
                          test_targets, test_targets_one_hot, test_class_names)

            self.net_train.fit(x=train_dataset, validation_data=None, epochs=epoch + self.evaluate_every, initial_epoch=epoch,
                         verbose=2, callbacks=callbacks)


    def validate(self, epoch, val_inputs, val_targets, val_targets_one_hot, val_class_names,
                 test_inputs, test_targets, test_targets_one_hot, test_class_names):

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

        util.metrics_to_csv(os.path.join(epoch_folder, "metrics_epoch_{}.csv".format(epoch)), np.asarray([val_accuracy,
                                                                                                          test_accuracy]),
                            ["siamese_val_accuracy", "siamese_test_accuracy"])

        if self.save_weights:
            self.net_train.save(os.path.join(self.results_path, "saved_model_train"), overwrite=True,
                                include_optimizer=False)
            self.net_train.save_weights(os.path.join(self.results_path, "saved_model_train/weights.h5"))
            self.net_test.save(os.path.join(self.results_path, "saved_model_test"), overwrite=True,
                                include_optimizer=False)
            self.net_test.save_weights(os.path.join(self.results_path, "saved_model_test/weights.h5"))

        if self.write_to_tensorboard:
            self._write_logs_to_tensorboard(epoch, val_accuracy, test_accuracy, test_probs_std, test_probs_means)

    def test(self, train_class_names, val_class_names, test_class_names, train_filenames,
             val_filenames, test_filenames, train_class_indices, val_class_indices,
             test_class_indices, num_val_samples, num_test_samples):
        num_train_cls = len(train_class_indices)
        self.setup_network(num_train_cls)
        val_inputs, val_targets, val_targets_one_hot = self.setup_input_test(val_class_indices, num_val_samples,
                                                                             val_filenames, 'val')
        test_inputs, test_targets, test_targets_one_hot = self.setup_input_test(test_class_indices, num_test_samples,
                                                                                test_filenames, 'test')

        val_accuracy, val_y_pred, val_predictions, val_probs_std, val_probs_means, mean_delay, std_delay = self.eval(
            val_inputs,
            val_targets,
            val_class_names)
        test_accuracy, test_y_pred, test_predictions, test_probs_std, test_probs_means, mean_delay, std_delay = self.eval(
            test_inputs,
            test_targets,
            test_class_names)

        util.metrics_to_csv(os.path.join(self.results_path, "metrics_inference.csv"), np.asarray([val_accuracy,
                                                                                                  test_accuracy,
                                                                                                  mean_delay,
                                                                                                  std_delay]),
                            ["siamese_val_accuracy", "siamese_test_accuracy", "mean_delay", "std_delay"])

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
            probs = self.net_train.predict([inps[0][trial], inps[1][trial]])[1]
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

        logger.info(
            "{} way one-shot accuracy: {}% on classes {}".format(self.num_val_ways, accuracy * 100, class_names))
        return accuracy, y_pred, preds, probs_std, probs_means, mean_delay, std_delay

    def _make_oneshot_task(self, n_val_tasks, image_data, labels, n_ways):
        classes = np.unique(labels)
        assert len(classes) == n_ways
        if len(image_data) < n_val_tasks:
            replace = True
        else:
            replace = False
        reference_indices = rng.choice(range(len(labels)), size=(n_val_tasks,), replace=replace)
        reference_labels = np.ravel(labels[reference_indices])
        comparison_indices = np.zeros((n_val_tasks, n_ways), dtype=np.int32)
        targets_one_hot = np.zeros((n_val_tasks, n_ways))
        targets_one_hot[range(n_val_tasks), reference_labels] = 1
        for i, cls in enumerate(classes):
            cls_indices = np.where(labels == cls)[0]
            comparison_indices[:, i] = rng.choice(cls_indices, size=(n_val_tasks,),
                                                  replace=True)
        comparison_images = image_data[comparison_indices, :, :, :]
        reference_images = image_data[reference_indices, np.newaxis, :, :, :]
        reference_images = np.repeat(reference_images, n_ways, axis=1)
        image_pairs = [np.array(reference_images, dtype=np.float32),
                       np.array(comparison_images, dtype=np.float32)]
        targets = np.argmax(targets_one_hot, axis=1)
        return image_pairs, targets, targets_one_hot

    def _write_logs_to_tensorboard(self, epoch, val_accuracy, test_accuracy, test_probs_std, test_probs_means):
        """ Writes the logs to a tensorflow log file
        """

        with self.summary_writer.as_default():
            tf.summary.scalar('Siamese validation accuracy', val_accuracy, step=epoch)
            tf.summary.scalar('Siamese test accuracy', test_accuracy, step=epoch)
