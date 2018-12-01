"""
File:   test.py
Author: Group w0w
Date:   2018/12/01
Desc:   Test the X for handwritten character recognition on an unknown test data
        set.
Refs:   -
        -
"""
import numpy as np
import pickle
import sys
import train


# TODO: Add command parsing.


def classifyImages(testImages, trainedModel):
    """

    :param testImages: processed test images
    :param trainedModel: trained model generated during training phase
    :return:
    """
    return 0


def preprocess(testImages):
    """
    Pre-process the test images to make them all of the same size.

    :param testImages: test images from provided npy file
    :return:
    """
    return 0


def main(inputFileName, useModel=True):
    """
    Run the model generation and/or the handwritten character recognition
    function.

    :param inputFileName: name of input file containing character images
	:param useModel: use pre-generated model for classification
    :return:
    """
    # If a pre-generated model is not used, then generate a new model from the
    # training data.
    if useModel:
        # TODO: Save the model in the genModel function as 'trained_model.pkl'.
        with open('trained_model.pkl', 'rb') as f:
            trainedModel = pickle.load(f)
    else:
        trainedModel = train.genModel()

    # Load test images
    testImages = np.load(inputFileName)

    # Pre-process the test images
    testImages = preprocess(testImages)

    # Label the characters in the images.
    labels = classifyImages(testImages, trainedModel)

    # Save the labels to an npy file
    np.save('outlabel', labels)


if __name__ == '__main__':
    # Name of npy file with test images.
    inputFileName = 'test_images.npy'

    main(inputFileName, useModel=False)

