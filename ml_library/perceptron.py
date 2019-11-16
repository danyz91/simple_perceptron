import numpy as np
from tqdm import tqdm

class Perceptron:
    '''
    The perceptron class represent one perceptron and grants to predict output or train the perceptron
    '''
    def __init__(self, weights_size, training_length=100, learning_rate=0.01):
        '''

        :param weights_size: dimension of weights array
        :param training_length: epochs number to be used during learning
        :param learning_rate: weights update learning rate
        '''
        self.weights = np.zeros(weights_size)
        self.bias = 0
        self.training_length = training_length
        self.learning_rate = learning_rate

    def predict(self, inputs):
        '''
        The function performs one prediction of the perceptron on given input. It implements w*x+b expression
        :param inputs:
        :return: activation value relative to given input
        '''

        perc_sum = np.dot(self.weights, inputs)+self.bias

        if perc_sum > 0:
            activation = 1
        else:
            activation = 0

        return activation

    def train(self, dataset):
        '''
        The function implements supervised learning routine on given dataset
        :param dataset: the dataset object on which performing training
        :return:
        '''
        for _ in tqdm(range(self.training_length)):

            for curr_input, curr_label in dataset:

                curr_prediction = self.predict(curr_input)
                curr_error = curr_label-curr_prediction
                # Learning
                self.weights += self.learning_rate * curr_error * curr_input
                self.bias += self.learning_rate*curr_error

