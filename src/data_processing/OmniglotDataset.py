import numpy as np
from scipy.misc import imread
import os
import argparse
from data_processing.dataset_utils import augment_dataset, images_to_tfrecord


class OmniglotDataset(object):
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
        data_path = "/mnt/data/datasets/omniglot/all_alphabets"
        train_data, train_labels, test_data, test_labels, class_names_dict = \
            self.load_symbols(data_path)

        self.x_train = np.expand_dims(train_data, 3)
        self.y_train = train_labels
        self.x_test = np.expand_dims(test_data, 3)
        self.y_test = test_labels
        self.class_names_dict = class_names_dict

    def load_symbols(self, path):
        # if data not already unzipped, unzip it.
        if not os.path.exists(path):
            print("unzipping")
            os.chdir(path)
            os.system("unzip {}".format(path + ".zip"))
        data = []
        labels = []
        curr_y = 0
        class_names_dict = {}
        # we load every alphabet seperately so we can isolate them later
        for alphabet in os.listdir(path):
            print("loading alphabet: " + alphabet)
            alphabet_path = os.path.join(path, alphabet)
            # every letter/category has it's own column in the array, so  load seperately
            for letter in os.listdir(alphabet_path):
                class_names_dict[curr_y] = alphabet + "_" + letter
                letter_path = os.path.join(alphabet_path, letter)
                category_images = []
                categ_labels = []
                for filename in os.listdir(letter_path):
                    image_path = os.path.join(letter_path, filename)
                    image = imread(image_path)
                    category_images.append(image)
                    categ_labels.append(curr_y)
                curr_y += 1
                data.append(np.stack(category_images))
                labels.append(categ_labels)

        labels = np.vstack(labels)
        data = np.stack(data)
        train_data = self.combine_dims(data[:, :16, :, :])
        test_data = self.combine_dims(data[:, 16:, :, :])
        train_labels = self.combine_dims(labels[:, :16])
        test_labels = self.combine_dims(labels[:, 16:])
        return train_data, train_labels, test_data, test_labels, class_names_dict

    def combine_dims(self, array):
        shapes = list(array.shape)
        combined = shapes[0] * shapes[1]
        return np.reshape(array, [combined] + shapes[2:])

    def data_to_tfrecords(self):
        images_to_tfrecord(self.tfrecs_path, self.x_train, self.y_train,
                           self.x_test, self.y_test, self.class_names_dict,
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
    loader = OmniglotDataset(args)
    loader.data_to_tfrecords()
    print("Omniglot dataset converted to tfRecords in {}".format(args.tfrecs_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", help="Path to the dataset",
                        default="/mnt/data/datasets/omniglot")
    parser.add_argument("--tfrecs_path", help="Path to where to save the tfrecords",
                        default="/mnt/data/datasets/siamese_cluster_data_new/omniglot")
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
                        default=32)
    parser.add_argument('--test_augment',
                        default=8)
    args = parser.parse_args()
    main(args)
