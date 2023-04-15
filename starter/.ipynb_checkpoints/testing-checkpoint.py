# TEST
# import necessary libraries
import argparse
from train_2 import *

# define the argument parser
parser = argparse.ArgumentParser()

# specify the required arguments
parser.add_argument('--train', type=str, default="train_data", required=False, help='path to the training data')
parser.add_argument('--test', type=str, default="test_data", required=False, help='path to the testing data')
parser.add_argument('--model-dir', type=str, default="model", required=False, help='directory to save the trained model')

# specify the optional arguments
parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1.0, help='learning rate')

# parse the arguments
args = parser.parse_args()

# call the main function of the script with the parsed arguments
main(args)
