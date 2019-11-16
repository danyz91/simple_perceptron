import numpy as np

class Dataset:
    '''
    The class represent one dataset for supervised learning applications. It stores input entries and their relative
    label
    '''
    def __init__(self, name, input_size):
        '''

        :param name: Mnemonic of the dataset
        :param input_size: Dimension of input entry of the dataset
        '''

        self.name = name
        self.input_size = input_size
        self.dataset = list()

    def append_to_dataset(self, inputs, label):
        '''
        The function appends one entry to the dataset. The entry is build as a tuple (inputs, label)
        :param inputs: np array of input_size, it represents one input entry in the dataset
        :param label: label corresponding to inputs specified
        :return:
        '''

        if not isinstance(inputs, (np.ndarray)):
            print('Error, you must feed dataset with numpy array!')
            return

        if inputs.size != self.input_size:
            print('Error, Input array must be of size', self.input_size, '!')
            exit(0)

        self.dataset.append((inputs, label))

    def get_dataset(self):
        '''
        The function returns the dataset object
        :return: the dataset object
        '''

        return self.dataset




