import numpy as np
import argparse
from keras.datasets import cifar100
from data_processing.dataset_utils import images_to_tfrecord


class Cifar100Dataset(object):
    def __init__(self, args):
        self.rotation_range = args.rotation_range
        self.width_shift_range = args.width_shift_range
        self.height_shift_range = args.height_shift_range
        self.brightness_range = args.brightness_range
        self.shear_range = args.shear_range
        self.zoom_range = args.zoom_range
        self.channel_shift_range = args.channel_shift_range
        self.fill_mode = args.fill_mode
        self.cval = args.cval
        self.horizontal_flip = args.horizontal_flip
        self.vertical_flip = args.vertical_flip
        self.dataset_path = args.data_path
        self.tfrecs_path = args.tfrecs_path
        self.train_augment = args.train_augment
        self.test_augment = args.test_augment
        self.load_data()

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        self.y_train = y_train
        self.y_test = y_test
        self.x_train = x_train.astype('float32')
        #self.x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0))
        self.x_test = x_test.astype('float32')
        #self.x_test = (x_test - x_test.mean(axis=0)) / (x_test.std(axis=0))

    def data_to_tfrecords(self):
        keys, values = range(100), [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm']
        class_names_dict = dict(zip(keys, values))
        images_to_tfrecord(self.tfrecs_path, self.x_train, self.y_train,
                           self.x_test, self.y_test, class_names_dict,
                           self.train_augment, self.test_augment,
                           rotation_range=self.rotation_range,
                           width_shift_range=self.width_shift_range,
                           height_shift_range=self.height_shift_range,
                           brightness_range=self.brightness_range,
                           shear_range=self.shear_range,
                           zoom_range=self.zoom_range,
                           channel_shift_range=self.channel_shift_range,
                           fill_mode=self.fill_mode,
                           cval=self.cval,
                           horizontal_flip=self.horizontal_flip,
                           vertical_flip=self.vertical_flip)


def main(args):
    loader = Cifar100Dataset(args)
    loader.data_to_tfrecords()
    print("Cifar100 dataset converted to tfRecords in {}".format(args.tfrecs_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", help="Path to the dataset",
                        default="/mnt/data/datasets/cifar100")
    parser.add_argument("--tfrecs_path", help="Path to where to save the tfrecords",
                        default="/mnt/data/datasets/siamese_cluster_data_new/cifar100")
    parser.add_argument('--rotation_range',
                        default=15.)
    parser.add_argument('--width_shift_range',
                        default=0.1)
    parser.add_argument('--height_shift_range',
                        default=0.1)
    parser.add_argument('--brightness_range',
                        default=None)
    parser.add_argument('--shear_range',
                        default=0.1)
    parser.add_argument('--zoom_range',
                        default=0.15)
    parser.add_argument('--channel_shift_range',
                        default=0.15)
    parser.add_argument('--fill_mode',
                        default='nearest')
    parser.add_argument('--cval',
                        default=0.)
    parser.add_argument('--horizontal_flip',
                        default=True)
    parser.add_argument('--vertical_flip',
                        default=False)
    parser.add_argument('--data_format',
                        default=None)
    parser.add_argument('--train_augment',
                        default=1000)
    parser.add_argument('--test_augment',
                        default=0)
    args = parser.parse_args()
    main(args)
