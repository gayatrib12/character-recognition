"""
File:   test.py
Author: Group w0w
Date:   2018/12/01
Desc:   Test pre-trained CNN.
Refs:   - https://nextjournal.com/gkoehler/pytorch-mnist
        -
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from skimage.transform import resize


# TODO: Comment out debugging code.
# TODO: Check that unknowns show up in predictions image.
# TODO: Change final run numbers used.


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def cnnTest(network, testLoader, predictedLabels):
    """
    CNN test function.

    :param network:
    :param testLoader:
    :param predictedLabels:
    :return: void
    """
    network.eval()

    with torch.no_grad():
        i = 0
        for data, target in testLoader:
            output = network(data)
            pred = output.data.max(1, keepdim=True)[1]
            predictedLabels[i] = pred
            i += 1


def preprocess(images, resizedShape=(28, 28), fileName=''):
    """
    Pre-process the test images to make them all of the same size and store them
    in a tensor formatted correctly for use in PyTorch.

    :param images: test images from provided npy file
    :param resizedShape: resized shape
    :param fileName: file name for npy file of image tensor
    :return: processed images as a tensor
    """
    # For all images, resize to same shape and leave as binary images
    # (but they are now type float instead of bool).
    img = images[0]
    img = resize(img, resizedShape, preserve_range=True).round()

    # Create an image tensor by adding a third dimension to the first image
    imgTensor = img[np.newaxis, :]

    # Iterate over the remaining images
    for img in (images[1:]):
        img = resize(img, resizedShape, preserve_range=True).round()
        img = img[np.newaxis, :]
        imgTensor = np.concatenate((imgTensor, img))

    imgTensor = imgTensor[np.newaxis, :]  # add a fourth dimension to the tensor

    # Swap the 3rd the 4th dimensions of tensor to get imgTensor into the NCHW format.
    # That is number of images, number of color channels, height of image, and
    # width of image.
    imgTensor = np.swapaxes(imgTensor, 0, 1)

    if len(fileName) > 0:
        np.save(fileName, imgTensor)

    return imgTensor


def classifyImages(inputFileName, outputFileName, doExtraCredit=False):
    """
    Classify handwritten characters using a pre-trained CNN model.

    :param inputFileName: name of input file containing test images
    :param outputFileName: name of output file with labels
    :param doExtraCredit: do extra credit part of the project flag
    :return: void
    """
    # Choose the right model from all of the runs for the 'a or b' test case
    # and the 'extra credit' test case.
    if doExtraCredit:
        runNumber = 21
    else:
        runNumber = 20

    # Create a character list
    charList = ['unknown', 'a', 'b', 'c', 'd', 'h', 'i', 'j', 'k']

    print("Testing in progress, this may take a couple of minutes...\n")

    # Load test images
    testImages = preprocess(np.load(inputFileName))
    testDataset = torch.utils.data.TensorDataset(torch.FloatTensor(testImages),
                                           torch.LongTensor(np.zeros(testImages.shape[0])).squeeze())
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1)

    # Load model parameters from json file
    with open('parameters_{}.json'.format(runNumber), 'r') as f:
        params = json.load(f)

    # Load the pre-trained CNN from saved files.
    loadedNetwork = Net()
    loadedOptimizer = optim.SGD(loadedNetwork.parameters(),
                                lr=params['learning_rate'],
                                momentum=params['momentum'])

    networkStateDict = torch.load("model_{}.pth".format(runNumber))
    loadedNetwork.load_state_dict(networkStateDict)

    optimizerStateDict = torch.load("optimizer_{}.pth".format(runNumber))
    loadedOptimizer.load_state_dict(optimizerStateDict)

    # Predict the labels in the test images using CNN.
    predictedLabels = torch.ones(testImages.shape[0])

    cnnTest(loadedNetwork, testLoader, predictedLabels)
    predictedLabels = predictedLabels.numpy().astype(int)

    mask = predictedLabels == 0
    predictedLabels[mask] = -1

    # Save the predicted labels to an npy file
    np.save(outputFileName, np.asarray(predictedLabels))

    # FOR DEBUGGING
    # Print out accuracy. Labels file must end in '_labels.npy'.
    labelsFile = inputFileName[:-4] + '_labels.npy'
    if os.path.isfile(labelsFile):
        groundTruth = np.load(labelsFile).astype(int).ravel()
        mask2 = groundTruth == predictedLabels
        print('Accuracy: {}%'.format(100. * np.sum(mask2) / len(groundTruth)))

    exampleIdx = np.random.choice(np.arange(testImages.shape[0]), 6, replace=False)

    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(testImages[exampleIdx[i], 0], cmap='gray', interpolation='none')
        tmpTarget = predictedLabels[exampleIdx[i]]
        if tmpTarget == 0:
            tmpTarget = -1
        plt.title("Prediction: {} ({})".format(tmpTarget,
            charList[predictedLabels[exampleIdx[i]]]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig('predictions.png')

    print("Example data plot has been saved as 'predictions.png'.")

if __name__ == '__main__':
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print('usage: python {} inData.npy out (-e)'.format(sys.argv[0]))
        sys.exit(0)

    inputFileName = sys.argv[1]
    outputFileName = sys.argv[2]

    if len(sys.argv) == 4:
        doExtraCredit = sys.argv[3] == '-e'
        if not doExtraCredit:
            print('usage: python {} inData.npy out (-e)'.format(sys.argv[0]))
            sys.exit(0)
    else:
        doExtraCredit = False

    # FOR DEBUGGING
    # Name of npy file with test images and labels.
    # 1st test - 'ab', 'original model', 150 epochs, 73.25%
    # inputFileName = 'testab.npy'
    # outputFileName = 'out'
    # doExtraCredit = False
    # 2nd test - 'all', 'original model', 150 epochs, 67.44%
    # inputFileName = 'testall.npy'
    # outputFileName = 'out'
    # doExtraCredit = True
    # 3rd test - 'ab', 'evenly sampled model', 300 epochs, 95.5%
    # inputFileName = 'testab.npy'
    # outputFileName = 'out'
    # doExtraCredit = False
    # 4th test - 'all', 'evenly sampledmodel', 300 epochs, 83.33%
    # inputFileName = 'testall.npy'
    # outputFileName = 'out'
    # doExtraCredit = True

    classifyImages(inputFileName, outputFileName, doExtraCredit)
