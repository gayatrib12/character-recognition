"""
File:   train.py
Author: Group w0w
Date:   2018/12/01
Desc:   Train and cross validate a(n) X to recognize handwritten characters.
Refs:   -
        -
"""
import numpy as np
import pickle


def genModel():
    """
    Generate the model for recognizing handwritten characters using the training
    images.

    :return:
    """
    trainedModel = 0

    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(trainedModel, f)

    return trainedModel
