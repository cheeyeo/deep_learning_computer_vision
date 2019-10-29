## Transfer Learning

Exercises from Chapter 18 of book "Deep Learning for Computer Vision"

## Notes

* Transfer learning involves using models trained on one problem as a starting point on a related problem

* Allows use of pre-trained models direcly, as feature extraction or via fine tuning

* Useful when you have small dataset; if you have a large dataset can train the pre-trained model directly or via fine tuning?

* Transder learning refers to a process where a model trained on one problem is used in some way on a second, related problem

* Technique in deep learning whereby one or more layers in a trained model as used in a new model trained on the problem of interest; one or more layers from trained model are then used in new model trained on problem of interest

* Benefit of decreasing the training time for a neural network and can result in lower generalization error

* Weights in re-used layers may be used as starting point in training process and adapted to new problem

* Useful in cases where there is a lot more labeled data used to train the pre-trained model than existing dataset and the structure of the datasets are similar

### Transfer Learning for image recognition

* Pre-trained models trained on large datasets of more than 1_000_000 images for over 1000 categories hence can detect generic features

* Models achieved high performance

* Model weights accessible to public use


### Usage patterns of pre-trained models

* Pre-trained model used as-is to classify new images

* Feature extractor; pre-proces images and extract relevant features

* Integerated Feature extractor: Pre-trained model integrated into new model but layers of pre-trained model frozen during training

* Weight initialization: Pre-trained model integrated into new model and layers of pre-trained model trained with new model at a lower learning rate



### Models for transfer learning
* VGG16
* VGG19
* GoogLeNet(Inception V3)
* ResNet50

## Extensions

* Use pre-trained VGG model to classify own images

* Experiment classifying photos using different pre-trained model such as ResNet

* Use pre-trained model to create feature vectors for all images in small image classification dataset and fit an SVM using the embeddings as input