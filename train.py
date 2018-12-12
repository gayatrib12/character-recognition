"""
File:   train.py
Author: Group w0w
Date:   2018/12/01
Desc:   Train and cross validate a CNN to classify handwritten characters.
Refs:   - https://nextjournal.com/gkoehler/pytorch-mnist
        -
"""
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.model_selection import train_test_split
from test import preprocess


# TODO: Save the CNN architecture parameters to JSON file
# TODO: Change the plot format

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


def genModel(runNumber, params):
    """
    Generate the model for recognizing handwritten characters using the training
    images.

    :param runNumber:
    :param params: parameters dictionary
    :return: trained model
    """
    # Create partial path string
    partialPath = os.path.join('npy_files', params['npy_files_prefix'])

    # Disable CUDA deep neural network library and fix the RNG reed for
    # repeatable results.
    torch.backends.cudnn.enabled = False
    torch.manual_seed(params['random_seed'])  # TODO: Remove for final code

    # Choose the right character list based on whether we are using all of the
    # data or just the a's and b's.
    if not params['npy_files_prefix'] in ['all', 'sampledall']:
        charList = ['', 'a', 'b', 'c', 'd', 'h', 'i', 'j', 'k']  # character list
    else:
        charList = ['unknown', 'a', 'b', 'c', 'd', 'h', 'i', 'j', 'k']

    # Path to the npy file with the preprocessed images.
    processedDataFileName = os.path.join('npy_files',
        params['npy_files_prefix'] + 'data_preprocessed.npy')

    # Assign the values in the parameters dictionary to variables.
    randomSeed = params['random_seed']
    resizedShape = (params['resized_shape'], params['resized_shape'])
    numEpochs = params['number_of_epochs']
    trainBatchSize = params['train_batch_size']
    testBatchSize = params['test_batch_size']
    learningRate = params['learning_rate']
    momentum = params['momentum']
    logInterval = params['log_interval']
    testSize = params['test_size']  # test size for train/test split

    """================ Load Training and Test Data ================"""

    # Load preprocessed training images from npy file or preprocess the images.
    if os.path.isfile(processedDataFileName):
        images = np.load(processedDataFileName)
    else:
        images = preprocess(np.load(partialPath + 'data.npy'),
                            resizedShape=resizedShape,
                            fileName=processedDataFileName)

    # Load the labels. Labels -1 are changed to 0 because PyTorch's `nll_loss`
    # function does not like labels less than 0.
    origLabels = np.load(partialPath + 'labels.npy')
    labels = copy.deepcopy(origLabels)
    labels[labels == -1] = 0
    imagesIdx = np.arange(len(images))

    # Perform a train/test split of training images.
    trainingImagesIdx, testImagesIdx, trainingLabels, testLabels \
        = train_test_split(imagesIdx, labels, test_size=testSize,
                           random_state=randomSeed)
    trainingImages = images[trainingImagesIdx]  # this is a tensor
    testImages = images[testImagesIdx]  # this is a tensor

    # Load the training data into PyTorch
    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(trainingImages),
                                           torch.LongTensor(trainingLabels).squeeze())
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=trainBatchSize,
                                               shuffle=True)

    # Load the test data into PyTorch
    testDataset = torch.utils.data.TensorDataset(torch.FloatTensor(testImages),
                                           torch.LongTensor(testLabels).squeeze())
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=testBatchSize,
                                              shuffle=True)

    """======== Construct Convolutional Neural Network ========"""

    # Create a nueral network class with the specified parameters
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learningRate,
                          momentum=momentum)

    """================ Train Convolutional Neural Network ================"""

    trainLosses = []
    trainCounter = []
    testLosses = []
    testCounter = [i * len(trainLoader.dataset) for i in range(numEpochs + 1)]

    # Test the network with the randomly initialized values.
    cnnTest(network, testLoader, testLosses, params)

    # Train the convolutional neural network over a number of epochs
    for epoch in range(1, numEpochs + 1):
        cnnTrain(epoch, network, trainLoader, optimizer, trainLosses, trainCounter,
              runNumber, logInterval)
        cnnTest(network, testLoader, testLosses, params)

    print("Displaying plots. You may need to close the plot window for the script to finish running.")

    plt.figure()
    plt.plot(trainCounter, trainLosses, color='blue')
    plt.scatter(testCounter, testLosses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.savefig(os.path.join('images', 'accuracy_{}.pdf'.format(runNumber)))
    plt.show()

    """======================== Example Data Test ========================"""

    # Visualize example data
    examples = enumerate(testLoader)
    batchIdx, (exampleData, exampleTargets) = next(examples)

    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(exampleData[i][0], cmap='gray', interpolation='none')
        tmpTarget = exampleTargets[i]
        if tmpTarget == 0:
            tmpTarget = -1
        plt.title("Ground Truth: {} ({})".format(tmpTarget,
            charList[exampleTargets[i]]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(os.path.join('images', 'ground_truth_{}.pdf'.format(runNumber)))
    plt.show()

    # See how the CNN performs on the example data
    with torch.no_grad():
        output = network(exampleData)

    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(exampleData[i][0], cmap='gray', interpolation='none')
        tmpTarget = output.data.max(1, keepdim=True)[1][i].item()
        if tmpTarget == 0:
            tmpTarget = -1
        plt.title("Prediction: {} ({})".format(tmpTarget,
            charList[output.data.max(1, keepdim=True)[1][i].item()]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(os.path.join('images', 'prediction_{}.pdf'.format(runNumber)))
    plt.show()

    """======== Save parameters to JSON file ========"""

    with open('parameters_{}.json'.format(runNumber), 'w') as f:
        json.dump(params, f, indent='\t')


def cnnTest(network, testLoader, testLosses, params):
    """
    CNN test function.

    :param network:
    :param testLoader:
    :param testLosses:
    :param params: parameters dictionary
    :return: void
    """
    network.eval()
    testLoss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testLoader:
            output = network(data)
            testLoss += F.nll_loss(output, target,
                                    size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    testLoss /= len(testLoader.dataset)
    testLosses.append(testLoss)
    print(
        '\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(testLoader.dataset),
            100. * correct / len(testLoader.dataset)))
    params.update({'final_accuracy': float(100. * correct / len(testLoader.dataset))})


def cnnTrain(epoch, network, trainLoader, optimizer, trainLosses, trainCounter,
          runNumber, logInterval):
    """
    CNN train function.

    :param epoch: current epoch number
    :param network:
    :param trainLoader:
    :param optimizer:
    :param trainLosses:
    :param trainCounter:
    :param runNumber:
    :param logInterval:
    :return: void
    """
    network.train()
    for batchIdx, (data, target) in enumerate(trainLoader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batchIdx % logInterval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batchIdx * len(data), len(trainLoader.dataset),
                       100. * batchIdx / len(trainLoader), loss.item()))
            trainLosses.append(loss.item())
            trainCounter.append(
                (batchIdx * 64) + (
                            (epoch - 1) * len(trainLoader.dataset)))
            torch.save(network.state_dict(), 'model_{}.pth'.format(runNumber))
            torch.save(optimizer.state_dict(), 'optimizer_{}.pth'.format(runNumber))


if __name__ == '__main__':
    """======================== Tunable Parameters ========================"""

    runNumber = 19

    params = {
        'npy_files_prefix': 'ab',
        'random_seed': 4,
        'resized_shape': 28,
        'number_of_epochs': 300,
        'learning_rate': 0.01,
        'momentum': 0.5,
        'log_interval': 10,
        'test_size': 0.33,
        'train_batch_size': 32,
        'test_batch_size': 64,
        'final_accuracy': -1,
    }

    # Generate the trained model
    genModel(runNumber, params)
