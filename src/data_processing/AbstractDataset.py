import numpy as np


class AbstractDataset():
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

    def load_data(self, normalization_type):
        pass

    def compile_dataset(self, images, labels, classes, new_labels):
        images_array = []
        labels_array = []
        indices = [[0, 0]]
        symbol_index = 0
        for idx, cls in enumerate(classes):
            curr_label_id = np.where(labels == cls)[0]
            images_array.extend(images[curr_label_id])
            labels_array.extend([new_labels[idx]] * len(curr_label_id))
            symbol_index += len(curr_label_id)
            indices[idx][1] = symbol_index - 1
            if idx < len(classes) - 1:
                indices.append([symbol_index, symbol_index])
        labels_array = np.vstack(labels_array)
        indices = np.asarray(indices)
        images_array = np.asarray(images_array, dtype=np.float32)
        if len(np.shape(images_array)) != 3:
            images_array = np.expand_dims(images_array, 3)
        return images_array, labels_array, indices

    def make_dataset_from_npy(self, npy_paths, labels):
        images_array = []
        labels_array = []
        indices = [[0, 0]]
        symbol_index = 0
        for s, symbol_path in enumerate(npy_paths):
            curr_label = labels[s]
            print("Loading " + symbol_path)
            symbol_data = np.load(symbol_path)
            images_array.extend(symbol_data)
            labels_array.extend([curr_label] * len(symbol_data))
            symbol_index += len(symbol_data)
            indices[s][1] = symbol_index - 1
            if s < len(npy_paths) - 1:
                indices.append([symbol_index, symbol_index])
        labels_array = np.vstack(labels_array)
        images_array = np.asarray(images_array, dtype=np.float32)
        indices = np.asarray(indices)
        return images_array, labels_array, indices
