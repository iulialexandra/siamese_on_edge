import numpy as np
import tensorflow as tf
import logging
import imageio
import os
import csv
import pandas as pd
import numpy.random as rng
from random import shuffle
from skimage.transform import resize
from data_processing.image_utils import ImageTransformer
from data_processing.image_utils import load_img, img_to_array


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def avi_to_frame_list(avi_filename, video_limit=-1, resize_scale=None):
    """Creates a list of frames starting from an AVI movie.

    Parameters
    ----------

    avi_filename: name of the AVI movie
    gray: if True, the resulting images are treated as grey images with only
          one channel. If False, the images have three channels.
    """
    logging.info('Loading {}'.format(avi_filename))
    try:
        vid = imageio.get_reader(avi_filename, 'ffmpeg')
    except IOError:
        logging.error("Could not load meta information for file %".format(avi_filename))
        return None
    data = [im for im in vid.iter_data()]
    if data is None:
        return
    else:
        shuffle(data)
        video_limit = min(len(data), video_limit)
        assert video_limit != 0, "The video limit is 0"
        data = data[:video_limit]
        expanded_data = [np.expand_dims(im[:, :, 0], 2) for im in data]
        if resize_scale is not None:
            expanded_data = [resize(im, resize_scale, preserve_range=True) for im in expanded_data]
        logging.info('Loaded frames from {}.'.format(avi_filename))
        return expanded_data


def images_to_tfrecord(save_path, train_data, train_labels,
                       test_data, test_labels, class_names_dict,
                       train_augment, test_augment, **kwargs):
    def save_tfrecs(cls_idx, mode, data, labels, augment):
        curr_label_id = np.where(labels == cls_idx)[0]
        images = data[curr_label_id]
        filename = "class_{}_{}.tfrecords".format(cls_idx, mode)
        if type(images[0]) not in [str, np.str_] and augment:
            images = augment_dataset(images, augment, **kwargs)

        tf_writer = tf.python_io.TFRecordWriter(os.path.join(save_path, filename))
        for image in images:
            if type(image) in [str, np.str_]:
                image = img_to_array(load_img(image))
            image_dims = np.shape(image)
            image_raw = image.astype(np.uint8)
            image_raw = image_raw.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_dims[-3]),
                'width': _int64_feature(image_dims[-2]),
                'depth': _int64_feature(image_dims[-1]),
                'label': _int64_feature(int(cls_idx)),
                'image_raw': _bytes_feature(image_raw)}))
            tf_writer.write(example.SerializeToString())
        tf_writer.close()
        return len(images), filename
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset_description = open(os.path.join(save_path, 'dataset_description.csv'), 'w')
    with dataset_description:
        fields = ["symbol", "train_file", "test_file", "train_num_samples", "test_num_samples"]
        csv_writer = csv.DictWriter(dataset_description, fieldnames=fields)
        csv_writer.writeheader()
        for cls_idx in sorted(list(class_names_dict.keys())):
            train_num, train_filename = save_tfrecs(cls_idx, "train",
                                                    train_data, train_labels,
                                                    train_augment)
            test_num, test_filename = save_tfrecs(cls_idx, "test",
                                                  test_data, test_labels,
                                                  test_augment)

            csv_writer.writerow({"symbol": class_names_dict[cls_idx],
                                 "train_file": train_filename,
                                 "test_file": test_filename,
                                 "train_num_samples": train_num,
                                 "test_num_samples": test_num})


def augment_dataset(data, limit_num, **kwargs):
    transformer = ImageTransformer(**kwargs)
    num_images = len(data)

    if limit_num == 0:
        augmented_data = [transformer.random_transform(im) for im in data]
        new_data = np.concatenate((data, np.asarray(augmented_data)), axis=0)
        np.random.shuffle(new_data)
        return data[:num_images]

    elif num_images >= limit_num:
        np.random.shuffle(data)
        data = data[:limit_num]
        augmented_data = [transformer.random_transform(im) for im in data]
        new_data = np.concatenate((data, np.asarray(augmented_data)), axis=0)
        np.random.shuffle(new_data)
        return data[:num_images]

    elif num_images < limit_num:
        gap = limit_num - num_images
        image_index = rng.choice(range(num_images), size=(gap,), replace=True)
        gap_data = [transformer.random_transform(data[idx]) for idx in image_index]
        new_data = np.concatenate((data, np.asarray(gap_data)), axis=0)
        np.random.shuffle(new_data)
        return new_data


def parser(record, new_labels_dict, image_dims, resize_dims):
    """It parses one tfrecord entry

    Args:
        record: image + label
    """

    def tf_repeat(tensor, repeats):
        """
        Args:

        input: A Tensor. 1-D or higher.
        repeats: A list. Number of repeat for each dimension, length must be the same as the number
                         of dimensions in input

        Returns:

        A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
        """
        with tf.variable_scope("repeat"):
            expanded_tensor = tf.expand_dims(tensor, -1)
            multiples = [1] + repeats
            tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
            repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
        return repeated_tensor

    with tf.device('/cpu:0'):
        features = tf.parse_single_example(record,
                                           features={
                                               'height': tf.FixedLenFeature([], tf.int64),
                                               'width': tf.FixedLenFeature([], tf.int64),
                                               'depth': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.int64),
                                           })

        label = tf.cast(features["label"], tf.int32)
        new_label = new_labels_dict.lookup(label)

        image_shape = tf.stack(list(image_dims))
        image = tf.decode_raw(features["image_raw"], tf.uint8)
        image = tf.cast(image, tf.float16)
        image = tf.scalar_mul(1 / (2 ** 8), image)
        image = tf.reshape(image, image_shape)

        if resize_dims is not None:
            image = tf.image.resize_images(
                image, resize_dims,
                align_corners=False,
                preserve_aspect_ratio=False)

    return image, new_label


def array_to_dataset(data_array, labels_array, batch_size):
    """Creates a tensorflow dataset starting from numpy arrays.
    NOTE: Not in use.
    """
    random_array = np.arange(len(labels_array))
    rng.shuffle(random_array)
    labels = labels_array[random_array]
    data_array = tf.cast(data_array[random_array], tf.float32)
    labels = tf.cast(labels, tf.int8)
    dataset = tf.data.Dataset.from_tensor_slices((data_array, labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset


def tfrecords_to_dataset(filenames, new_labels_dict, batch_size, im_size, shuffle, re_size=None):
    """ Starting from tfRecord files, it creates tensorflow datasets.
    """
    with tf.device('/cpu:0'):
        files = tf.data.Dataset.list_files(filenames)
        if shuffle:
            dataset = files.apply(tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=len(filenames),
                block_length=max(2, batch_size // 4)))

        else:
            dataset = files.apply(tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=len(filenames),
                block_length=max(2, batch_size // len(filenames))))

        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(map_func=lambda x: parser(x, new_labels_dict, im_size,
                                                                    re_size),
                                          batch_size=batch_size,
                                          num_parallel_calls=4, drop_remainder=True))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10)
        dataset.prefetch(buffer_size=batch_size * 10)
    return dataset


def deploy_dataset(filenames, new_labels_dict, batch_size, image_dims, shuffle=True):
    dataset = tfrecords_to_dataset(filenames, new_labels_dict, batch_size, image_dims, shuffle)
    iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()
    labels_one_hot = tf.one_hot(label_batch, len(filenames))
    return iterator, image_batch, label_batch, labels_one_hot


def sample_class_images(data, labels):
    unique_labels = np.unique(labels)
    sampled_data = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        im = data[indices[0]]
        sampled_data.append(im)
    return sampled_data, unique_labels


def read_dataset_csv(dataset_path, val_ways):
    def join_paths(class_index, phase_data):
        return os.path.join(dataset_path, "class_{}_{}.tfrecords".format(class_index, phase_data))

    dataset = pd.read_csv(os.path.join(dataset_path, "dataset_description.csv"))
    train_names = dataset["symbol"].values
    val_names = train_names.copy()
    train_num_samples = dataset["train_num_samples"].values
    val_samples = dataset["test_num_samples"].values
    train_indices = np.arange(len(train_names))
    # train data
    np.random.shuffle(train_indices)
    train_class_indices = train_indices[:-val_ways]
    train_class_names = train_names[train_class_indices]
    train_filenames = [join_paths(i, "train") for i in train_class_indices]

    val_class_indices = np.random.choice(train_class_indices, val_ways, replace=False)
    val_class_names = val_names[val_class_indices]
    val_num_samples = val_samples[val_class_indices]
    val_filenames = [join_paths(i, "test") for i in val_class_indices]
    num_val_samples = sum(val_num_samples)

    # test data
    test_class_indices = train_indices[-val_ways:]
    test_class_names = train_names[test_class_indices]
    test_filenames = [join_paths(i, "train") for i in test_class_indices]
    test_num_samples = train_num_samples[test_class_indices]
    num_test_samples = sum(test_num_samples)

    return train_class_names, val_class_names, test_class_names, train_filenames, val_filenames, \
           test_filenames, train_class_indices, val_class_indices, test_class_indices, \
           num_val_samples, num_test_samples
