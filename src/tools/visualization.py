import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import itertools
import tensorflow as tf
from skimage.transform import resize


def plot_validation_images(plots_path, images, target):
    width, height, ch = images[0][0].shape
    fig, axes = plt.subplots(len(target), 1)
    fig.set_size_inches(25, 10)

    for i in range(len(target)):
        ax = axes[i]
        image = np.concatenate((images[0][i], np.zeros((width, 10, ch)), images[1][i]), axis=1)
        if target[i] == 0:
            ax.set_title("Different class", fontsize=20)
        else:
            ax.set_title("Same class", fontsize=20)
        ax.imshow(np.array(np.squeeze(image), np.float32), cmap="gist_gray")
        ax.set_axis_off()
    fig.savefig(plots_path, bbox_inches='tight', dpi=300)
    plt.close("all")


def plot_wrong_preds(plots_path, images, target, prediction):
    width, height, ch = images[0][0].shape
    fig, axes = plt.subplots(len(target), 1)
    fig.set_size_inches(25, 10)
    titles = ["same" if prediction == i else "different" for i in range(len(target))]
    for i in range(len(target)):
        ax = axes[i]
        image = np.concatenate((images[0][i], np.zeros((width, 10, ch)), images[1][i]), axis=1)
        if target[i] == 0:
            ax.set_title("Different class, predicted {}".format(titles[i]), fontsize=20)
        else:
            ax.set_title("Same class, predicted {}".format(titles[i]), fontsize=20)
        ax.imshow(np.array(np.squeeze(image), np.float32), cmap="gist_gray")
        ax.set_axis_off()
    fig.savefig(plots_path, bbox_inches='tight', dpi=300)
    plt.close("all")


def plot_siamese_training_pairs(plots_path, images, target):
    width, height, ch = images[0].shape
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(25, 10)

    image = np.concatenate((images[0], np.zeros((width, 10, ch)), images[1]), axis=1)
    if target == 0:
        ax.set_title("Different class", fontsize=20)
        path = plots_path + "_diff.png"
    else:
        ax.set_title("Same class", fontsize=20)
        path = plots_path + "_same.png"
    ax.imshow(np.array(np.squeeze(image), np.float32), cmap="gist_gray")
    ax.set_axis_off()
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close("all")


def plot_classification_images(plot_path, input_samples, labels, type):
    num_images = len(labels)
    num_cols = 5
    num_rows = int(np.ceil(num_images / num_cols))

    fig_title = plot_path.split("/")[-1]
    fig, ax = plt.subplots(num_rows, num_cols)
    fig.set_size_inches(25, 25)
    fig.suptitle(fig_title, fontsize=35)

    for i in range(num_rows):
        for j in range(num_cols):
            ax[i][j].imshow(np.squeeze(input_samples[i * num_rows + j]), cmap="gray")
            ax[i][j].set_axis_off()
            ax[i][j].set_title("Class {}".format(str(labels[i * num_rows + j])), fontsize=30)
    plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
    fig.savefig(os.path.join(plot_path, "Classification {} images".format(type)),
                bbox_inches='tight', dpi=300)
    plt.close("all")


def get_layer_outputs(sample, input_layer, output_layer):
    get_output = tf.keras.backend.function([input_layer.input], [output_layer.output])
    layer_output = get_output([sample[np.newaxis, :]])
    return layer_output


def plot_siamese_features(path, type, input_batch, targets, model):
    def get_branch_outputs(image_pair, layer):
        image_left = image_pair[0]
        image_right = image_pair[1]

        l_output = get_layer_outputs(image_left, model.layers[2].layers[0],
                                     layer)
        r_output = get_layer_outputs(image_right, model.layers[2].layers[0],
                                     layer)
        l_branch = np.squeeze(l_output)
        r_branch = np.squeeze(r_output)
        return l_branch, r_branch

    fig_title = path.split("/")[-1]
    chosen_layers = [layer for layer in model.layers[2].layers if "conv" in str(layer)]
    for l, layer in enumerate(chosen_layers):
        left_features, right_features = get_branch_outputs([input_batch[0][0],
                                                            input_batch[1][0]],
                                                           layer)
        if len(np.shape(left_features)) != 3:
            continue
        plot_path = os.path.join(path, layer.name)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for t, target in enumerate(targets):
            left_image = input_batch[0][t]
            right_image = input_batch[1][t]
            left_features, right_features = get_branch_outputs([left_image, right_image], layer)
            f_map_width, f_map_height, f_map_channels = left_features.shape
            max_to_plot = min(f_map_channels, 19)
            map_idx_to_plot = np.linspace(start=0, stop=f_map_channels, num=max_to_plot, dtype=int,
                                          endpoint=False)
            left_maps_to_plot = left_features[:, :, map_idx_to_plot]
            right_maps_to_plot = right_features[:, :, map_idx_to_plot]

            num_cols = 5
            num_rows = int(np.ceil((max_to_plot + 1) / num_cols))

            fig, ax = plt.subplots(num_rows, num_cols * 2)
            [axi.set_axis_off() for axi in ax.ravel()]

            fig.set_size_inches(40, 40)
            fig.suptitle(fig_title, fontsize=35)

            ax[0][0].imshow(np.squeeze(left_image))
            ax[0][0].set_axis_off()
            ax[0][5].imshow(np.squeeze(right_image))
            ax[0][5].set_axis_off()

            idx = 0
            for i in range(num_rows):
                if idx == max_to_plot:
                    break
                if i == 0:
                    r = range(1, num_cols)
                else:
                    r = range(num_cols)
                for j in r:
                    if idx == max_to_plot:
                        break
                    ax[i][j].imshow(np.squeeze(left_maps_to_plot[:, :, idx]), cmap="gray")
                    ax[i][j + 5].imshow(np.squeeze(right_maps_to_plot[:, :, idx]), cmap="gray")
                    idx += 1

            plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
            fig.savefig(os.path.join(plot_path, "{}_feature_maps_im_{}.png".format(str(type, t))),
                        bbox_inches='tight', dpi=300)
            plt.close("all")


def stitch_feature_maps(input_sample, feature_maps):
    f_map_width, f_map_height, f_map_channels = feature_maps.shape
    max_to_plot = min(f_map_channels, 19)
    map_idx_to_plot = np.linspace(start=0, stop=f_map_channels, num=max_to_plot, dtype=int,
                                  endpoint=False)
    maps_to_plot = feature_maps[:, :, map_idx_to_plot]
    resized_image = resize(input_sample, (f_map_width, f_map_height,
                                          np.shape(input_sample)[-1]))
    plot_images = np.concatenate((np.mean(resized_image, axis=-1, keepdims=True), maps_to_plot),
                                 axis=2)
    num_subimages = np.shape(plot_images)[-1]
    num_rows = int(np.ceil(num_subimages / 5))
    num_cols = 5
    margin = max(1, f_map_width // 5)
    canvas_width = num_cols * f_map_width + (num_cols - 1) * margin
    canvas_height = num_rows * f_map_height + (num_rows - 1) * margin
    stitched_features = np.zeros((canvas_width, canvas_height, 1))

    # fill the picture with our saved filters
    counter = 0
    for i in range(num_cols):
        for j in range(num_rows):
            if counter < num_subimages:
                map_index = i * num_rows + j
                img = plot_images[:, :, map_index, np.newaxis]
                width_margin = (f_map_width + margin) * i
                height_margin = (f_map_height + margin) * j
                stitched_features[width_margin: width_margin + f_map_width,
                height_margin: height_margin + f_map_height, :] = img
                counter += 1
    return stitched_features


def plot_weights(path, model, type):
    fig_title = path.split("/")[-1]

    if type == "classifier":
        level = model.layers[1].layers
    else:
        level = model.layers[2].layers
    chosen_layers = [layer for layer in level if "conv2d" in layer.name]
    for num_layer, layer in enumerate(chosen_layers):
        plot_path = os.path.join(path, layer.name)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        weights = layer.get_weights()[0]
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(25, 25)
        fig.suptitle(fig_title, fontsize=35)

        sns.distplot(np.ravel(weights), ax=ax)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.tick_params(axis='both', which='minor', labelsize=25)
        fig.savefig(os.path.join(plot_path, "weights.png"),
                    bbox_inches='tight', dpi=300)
        plt.close("all")


def plot_confusion_matrix(plot_path, type, confusion_matrix, classes, cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig_title = plot_path.split("/")[-1]

    def plot(cm, title, fmt):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title + fig_title, fontsize=12)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "{}_{}.png".format(type, title)),
                    bbox_inches='tight', dpi=300)
        plt.close("all")

    # plot conf matrix without normalization
    title = "Un-normalized confusion matrix "
    fmt = 'd'
    plot(confusion_matrix, title, fmt)

    normalized_cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    title = "Normalized confusion matrix "
    fmt = '.2f'
    plot(normalized_cm, title, fmt)


def plot_pred_diffs(path, type, pred_diffs):
    fig_title = path.split("/")[-1]
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(25, 25)
    sns.distplot(pred_diffs, ax=ax)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.tick_params(axis='both', which='minor', labelsize=25)
    plt.title(fig_title, fontsize=30)
    fig.savefig(os.path.join(path, "{}_prediction_differences.png".format({type})),
                bbox_inches='tight', dpi=300)
    plt.close("all")
