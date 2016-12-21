---
title: "Kaggle - MNIST Digit Recognizer"
author: "Sameh Awaida"
date: "12/21/2016"
output: html_document
---


# Classify handwritten digits using the famous MNIST data


The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is. 

For more details on this competition, see <https://www.kaggle.com/c/digit-recognizer>.

## Classifiers:

- [Using R Random Forests](https://github.com/sameha/Kaggle---MNIST-Digit-Recognizer/blob/master/r_digit_recognizer_random_forest.Rmd)
- [Using Torch7 CNN](https://github.com/sameha/Kaggle---MNIST-Digit-Recognizer/blob/master/itorch_digit_recognizer_cnn.md)

## Results:


The tutorial accuracy results are:


- CNN 40 training loops with an accuracy of 99.429% for a 113th position in the Kaggle competition.
- CNN 20 training loops with an accuracy of 99.414% for a 114 position in the Kaggle competition.
- CNN 10 training loops with an accuracy of 99.286% for a 148 position in the Kaggle competition.
- Random Forests using R with an accuracy of 95.84%.

## Competition Data:


The data for this competition were taken from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/index.html). The MNIST ("Modified National Institute of Standards and Technology") dataset is a classic within the Machine Learning community that has been extensively studied.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.

## Overview of the dataset:


- 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 0.
- 42000 digits for training, 28000 digits for testing.
- The inputs (images) are 1x28x28 centered around a single digit.
- The outputs (targets) are 10-dimensional
