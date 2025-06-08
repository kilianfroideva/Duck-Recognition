# ARN Duck Species Classification Project

## Introduction

The goal of this project is to classify images of ducks into different species using a Convolutional Neural Network (CNN). The dataset consists of images of ducks taken in various locations and the classification task is to identify the species of each duck.

The idea is that the user can take a picture of a duck with their phone (bad quality picture) and the model will classify it into one of the species (on a predefined list). The model should be lightweight enough to run on a mobile device, so we will use transfer learning with MobileNetV2.

## The Problem

The dataset consists of images of ducks taken in various locations with the phone and taken by us. The species to classify are:

- `Canard colvert mâle`
- `Canard colvert femelle`
- `Foulque macroule`
- `Grèbe huppé`
- `Autre`

The first idea was to classify the species of ducks that are commonly found in Switzerland. The dataset is small, with only a few images per species, which makes it a challenging task.

Initially, we wanted to classify more species of ducks (like `Cygne Tuberculé` and `Harle Bièvre`), but due to the limited number of images available, we had to reduce the number of species to five. The dataset is unbalanced, with some species having more images than others.

We will still test the model performance on these 2 more species to see it's performance but since we had a hard time to find a working model, we decided to focus on the five species listed above.

(TODO: add a histogram of classes)

One thing to note is that for duck species with different features between sexes, we split them into two classes: `Canard colvert mâle` and `Canard colvert femelle` to help the model distinguish between them (it also give one more feature to the application).

## Data Preparation

### Data query

The dataset was created by taking pictures of ducks in various locations. The images were taken with a phone camera, which means that the quality of the images is not always optimal. The images were taken in various locations and conditions, which adds to the complexity of the classification task.

We croped the images with multiple ducks in the same picture to focus on the duck we want to classify.

We also took pictures of other species and backround without ducks to help the model distinguish between the species and the background so that it could classify these images as `Autre`.

### Data preprocessing

We resized the images to a fixed size of 224x224 pixels to match the input size of the MobileNetV2 model. We also normalized the pixel values to be in the range [0, 1] by dividing by 255.

### Data Augmentation

Since the dataset is small, we used data augmentation techniques to artificially increase the size of the dataset. We applied most of data augmentation techniques that we found of such as:
- Random rotation
- Random horizontal flip
- Random brightness adjustment
- Random contrast adjustment
- And more (TODO: list all the data augmentation techniques used)

We also have augmented each class so that at end, we have a balanced dataset.

## Model Creation

We used transfer learning with MobileNetV2 as the base model. The model was trained on the ImageNet dataset, which means that it has already learned to recognize a wide variety of objects. We added a few layers on top of the base model to adapt it to our specific classification task.

The first thing we realized is that we want to maximize the macro f1-score, so we will use the `f1_score` as the metric to optimize during training. We will also use the `categorical_crossentropy` loss function since we have multiple classes to classify.

We also choose to use the `Adam` optimizer since it is a good optimizer for fast results.

TODO: how to choose the hyperparameters (number of epochs, learning rate, etc.)?

### First working model

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ mobilenetv2_1.00_224 (Functional)    │ (None, 7, 7, 1280)          │       2,257,984 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 1280)                │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 1280)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │         163,968 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 5)                   │             645 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘


## Results

### Model testing

- Verify if with totally new duck species, the model can classify it as `Autre` (Other) or not. We hope that the model will not classify it as one of the species it was trained on.
- Multiple ducks in the same picture:
  1. We hope that the model could classify the most prominent duck in the picture, but we are not sure if it will work or not.
  2. If there is one known species and one unknown species, we hope that the model will classify the known species.



## Conclusions

### To go further

- Not trying to place the duck in different place on the image since with that, we could do data augmentation by zooming. Since we have taken pictures near the border of the picture, we couldn't do data augmentation by zooming (that could give more images to working with since we already have a not much to work with) and notify the user to place the duck in the middle of the picture.

- Lack of pictures for more species especially for `Cygne` and `Harle Bièvre` but that helped to do the analysis of the model since it could help to see if it really can distinguish the species or not. If a new species of duck is tested, it will be classified as `Autre` (Other) since the model is not trained to recognize it (TODO we hope/verify) 

- Multiple ducks in the same pictures classification
- More duck species
- Lack of diversity in the dataset, all images were taken in similar locations and conditions. This could affect the model's ability to generalize to new images taken in different conditions.
