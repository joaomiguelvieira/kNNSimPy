import argparse
from Dataset import Dataset
from KNNAlgorithm import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('n_train', type=int, help="Number of training samples")
    parser.add_argument('n_test', type=int, help="Number of testing samples")
    parser.add_argument('n_features', type=int, help="Number of features per sample")
    parser.add_argument('n_classes', type=int, help="Number of classes")
    parser.add_argument('k', type=int, help="Number of nearest neighbors")
    parser.add_argument('--input-file', type=str, help="Binary input file", default=None)
    parser.add_argument('--solution-file', type=str, help="Solution file", default=None)
    parser.add_argument('--data-type', choices=['numpy', 'torch'], default='numpy')

    args = parser.parse_args()

    dataset = Dataset(args.input_file, args.n_train, args.n_test, args.n_features, args.n_classes, args.data_type)
    classify(dataset, args.k)

    if args.solution_file is not None:
        errors = check_solution(dataset, args.solution_file)
        if errors != 0:
            print(f"{errors} classification errors found!")
        else:
            print("No errors found!")


if __name__ == '__main__':
    main()
