import numpy as np
import argparse

from ml_library.perceptron import Perceptron
from ml_library.dataset_collection import DatasetCollection

parser = argparse.ArgumentParser(description='Simple Binary Classification Example')

parser.add_argument('-d', '--dataset', type=str, help='dataset name', required=True)
parser.add_argument('-t', '--training_length', type=int, help='perceptron training length', default=100)
parser.add_argument('-r', '--learning_rate', type=float, help='perceptron learning rate', default=0.1)


def main():

    args = parser.parse_args()
    ds_collection = DatasetCollection()

    # Dataset param reading and handling
    selected_dataset = ds_collection.search_dataset_by_name(args.dataset)

    if selected_dataset is None:
        print('Dataset not found!')
        print('Possible names are')
        print([ds.name for ds in ds_collection.datasets])
        return

    # Perceptron building
    perceptron = Perceptron(weights_size=2, training_length=args.training_length, learning_rate=args.learning_rate)

    # Training
    perceptron.train(selected_dataset.dataset)
    print('Training Complete!')

    # Test
    tests = list()
    tests.append(np.array([0, 0]))
    tests.append(np.array([0, 1]))
    tests.append(np.array([1, 0]))
    tests.append(np.array([1, 1]))

    for curr_test in tests:
        curr_output = perceptron.predict(curr_test)
        print('Input: ', curr_test, '-> Predicted :', curr_output)


if __name__=='__main__':
    main()
