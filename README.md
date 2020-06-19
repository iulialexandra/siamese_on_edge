# siamese_on_edge_tf2

This repository is similar to the improved_siamese one, with some additional code refactoring.
Working Tensorflow version is 2.1.0. The code is divided into training and testing. Testing performs additional time
 computations for an entire n-way siamese model. The net_test is not used for validation and testing within the code
 , but it's saved for inference speed computation without the 2 side classifiers. 