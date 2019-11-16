import numpy as np
from . import dataset


class DatasetCollection:
    '''
    Dataset collection for the library. It stores a list of dataset to be used in supervised learning
    '''
    def __init__(self):

        self.datasets=[]

        and_dataset = dataset.Dataset('AND', 2)
        and_dataset.append_to_dataset(np.array([0, 0]), 0)
        and_dataset.append_to_dataset(np.array([0, 1]), 0)
        and_dataset.append_to_dataset(np.array([1, 0]), 0)
        and_dataset.append_to_dataset(np.array([1, 1]), 1)

        self.datasets.append(and_dataset)

        or_dataset = dataset.Dataset('OR', 2)
        or_dataset.append_to_dataset(np.array([0, 0]), 0)
        or_dataset.append_to_dataset(np.array([0, 1]), 1)
        or_dataset.append_to_dataset(np.array([1, 0]), 1)
        or_dataset.append_to_dataset(np.array([1, 1]), 1)

        self.datasets.append(or_dataset)

        nand_dataset = dataset.Dataset('NAND', 2)
        nand_dataset.append_to_dataset(np.array([0, 0]), 1)
        nand_dataset.append_to_dataset(np.array([0, 1]), 1)
        nand_dataset.append_to_dataset(np.array([1, 0]), 1)
        nand_dataset.append_to_dataset(np.array([1, 1]), 0)

        self.datasets.append(nand_dataset)

        nor_dataset = dataset.Dataset('NOR', 2)
        nor_dataset.append_to_dataset(np.array([0, 0]), 1)
        nor_dataset.append_to_dataset(np.array([0, 1]), 0)
        nor_dataset.append_to_dataset(np.array([1, 0]), 0)
        nor_dataset.append_to_dataset(np.array([1, 1]), 0)

        self.datasets.append(nor_dataset)

    def search_dataset_by_name(self, name):
        '''
        The function returns dataset object relative to given name. None if dataset name is not present in the collection
        :param name: name of dataset to be searched
        :return:
        '''

        for ind, ds in enumerate(self.datasets):
            if name == ds.name:
                return self.datasets[ind]

        return None

