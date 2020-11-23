import glob
import re
import argparse
import random
import numpy as np
from data_processing.dataset_utils import augment_dataset, images_to_tfrecord


class ImagenetDataset(object):
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
        self.dataset_path = args.path
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
        label_dict, class_description = self.build_label_dicts()
        filenames_train = []
        labels_train = []
        filenames = glob.glob('/mnt/Storage/datasets/tiny-imagenet/train/*/images/*.JPEG')
        for filename in filenames:
            match = re.search(r'n\d+', filename)
            label = int(label_dict[match.group()])
            filenames_train.append(filename)
            labels_train.append(label)
        filenames_val = []
        labels_val = []
        with open('/mnt/Storage/datasets/tiny-imagenet/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                filename = '/mnt/Storage/datasets/tiny-imagenet/val/images/' + split_line[0]
                label = int(label_dict[split_line[1]])
                filenames_val.append(filename)
                labels_val.append(label)
        self.x_train = np.asarray(filenames_train)
        self.y_train = np.asarray(labels_train)
        self.x_test = np.asarray(filenames_val)
        self.y_test = np.asarray(labels_val)
        self.class_names_dict = class_description

    def build_label_dicts(self):
        """Build look-up dictionaries for class label, and class description
        Class labels are 0 to 199 in the same order as
          tiny-imagenet/wnids.txt. Class text descriptions are from
          tiny-imagenet/words.txt
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
        with open('/mnt/Storage/datasets/tiny-imagenet/wnids.txt', 'r') as f:
            for i, line in enumerate(f.readlines()):
                synset = line[:-1]  # remove \n
                label_dict[synset] = i
        with open('/mnt/Storage/datasets/tiny-imagenet/words.txt', 'r') as f:
            for i, line in enumerate(f.readlines()):
                synset, desc = line.split('\t')
                desc = desc[:-1]  # remove \n
                if synset in label_dict:
                    class_description[label_dict[synset]] = desc
        return label_dict, class_description

    def data_to_tfrecords(self):
        images_to_tfrecord(self.dataset_path, self.x_train, self.y_train, self.class_names_dict,
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
        images_to_tfrecord(self.dataset_path, self.x_test, self.y_test, self.class_names_dict,
                           "test", False, 0)


def main(args):
    loader = ImagenetDataset(args)
    loader.data_to_tfrecords()
    print("Imagenet dataset converted to tfRecords in {}".format(args.path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to mnist dataset",
                        default="/mnt/Storage/datasets/tiny-imagenet")
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
                        default=1000)
    args = parser.parse_args()
    main(args)
