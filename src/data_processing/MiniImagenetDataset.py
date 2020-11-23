import glob
import re
import argparse
import os
import random
import numpy as np
from data_processing.dataset_utils import augment_dataset, images_to_tfrecord


class MiniImagenetDataset(object):
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
        self.annot_path = args.annot_path
        self.ending = args.ending
        self.augment = args.augment
        self.augmentation_limit = args.augmentation_limit
        self.load_data()

    def load_data(self):
        """Gets filenames and labels
        Args:
          mode: 'train' or 'val'
            (Directory structure and file naming different for
            train and val datasets)
        Returns:
          list of tuples: (jpeg filename with path, label)
        """

        def analyze_folders(root_folder, folders, label_dict):
            filenames = []
            labels = []
            for i, folder in enumerate(folders):
                label = int(label_dict[folder])
                class_folder = os.path.join(root_folder, folder)
                for file in os.listdir(class_folder):
                    if file.endswith(self.ending):
                        filenames.append(os.path.join(class_folder, file))
                        labels.append(label)
            return np.asarray(filenames), np.asarray(labels)

        train_folder = os.path.join(self.dataset_path, 'train')
        test_folder = os.path.join(self.dataset_path, 'test')

        train_classes = [label for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))]
        test_classes = [label for label in os.listdir(test_folder)
                        if os.path.isdir(os.path.join(test_folder, label))]

        label_dict_train, self.class_description_train = self.build_label_dicts(train_classes)
        self.x_train, self.y_train = analyze_folders(train_folder, train_classes, label_dict_train)
        label_dict_test, self.class_description_test = self.build_label_dicts(test_classes)
        self.x_test, self.y_test = analyze_folders(test_folder, test_classes, label_dict_test)

    def build_label_dicts(self, train_synsets):
        """Build look-up dictionaries for class label, and class description
        Class labels are 0 to 199 in the same order as
        Returns:
          tuple of dicts
            label_dict:
              keys = synset (e.g. "n01944390")
              values = class integer {0 .. 199}
            class_desc:
              keys = class integer {0 .. 199}
              values = text description from words.txt
        """
        label_dict, class_description = {}, {}
        for i, synset in enumerate(train_synsets):
            label_dict[synset] = i
        with open(self.annot_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                synset, _, desc = line.split(' ')
                desc = desc[:-1]  # remove \n
                if synset in label_dict.keys():
                    class_description[label_dict[synset]] = desc
        return label_dict, class_description

    def data_to_tfrecords(self):
        images_to_tfrecord(self.tfrecs_path, self.x_train, self.y_train,
                           self.class_description_train,
                           "train", self.augment, self.augmentation_limit,
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
        images_to_tfrecord(self.tfrecs_path, self.x_test, self.y_test, self.class_description_test,
                           "test", False, 0)


def main(args):
    loader = MiniImagenetDataset(args)
    loader.data_to_tfrecords()
    print("Imagenet dataset converted to tfRecords in {}".format(args.tfrecs_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the dataset",
                        default="/media/iulialexandra/data/mini-imagenet-simclrfeatures/mini_r101_1x_sk0")
    parser.add_argument("--tfrecs_path", help="Path to where to save the tfrecords",
                        default="/media/iulialexandra/data/siamese_data_results/tfrecs/mini-simclr-r101")
    parser.add_argument("--annot_path", help="Path to where the Imagenet annotations are",
                        default="/media/iulialexandra/data/imagenet/ILSVRC2015/devkit/data/map_clsloc.txt")
    parser.add_argument("--ending", help="The file extension to take into consideration",
                        default=".npy")
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
    parser.add_argument('--augment',
                        default=False)
    parser.add_argument('--augmentation_limit',
                        default=0)
    args = parser.parse_args()
    main(args)