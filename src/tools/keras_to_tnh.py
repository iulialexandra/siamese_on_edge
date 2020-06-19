path = r"C:\Users\Alessandro\Google Drive\Projects\siamese_on_edge_tf2\results\75 sparsity 2020_6_18-17_36_21_808974_seed_13_tiny-imagenet_HorizontalNetworkV44_quantization_nullhop_classif_yes\saved_model_train"

from tensorflow.keras.models import load_model
model = load_model(path)