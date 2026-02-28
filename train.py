import argparse
from enum import Enum

from md_lstm.pipeline import main as train_md_lstm
from single_lstm.pipeline import train as train_single_lstm
from vanilla_lstm.binary_labeling import train as train_vanilla_binary_labeling
from vanilla_lstm.sequence_recall import train as train_vanilla_sequence_recall


class Model(str, Enum):
    MD_LSTM = 'md_lstm'
    SINGLE_LSTM = 'single_lstm'
    VANILLA_SEQUENCE_RECALL = 'vanilla_sequence_recall'
    VANILLA_BINARY_LABELING = 'vanilla_binary_labeling'


TRAIN_FUNCTIONS = {
    Model.MD_LSTM: train_md_lstm,
    Model.SINGLE_LSTM: train_single_lstm,
    Model.VANILLA_SEQUENCE_RECALL: train_vanilla_sequence_recall,
    Model.VANILLA_BINARY_LABELING: train_vanilla_binary_labeling,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=Model, required=True, choices=list(Model))
    args = parser.parse_args()

    TRAIN_FUNCTIONS[args.model]()
