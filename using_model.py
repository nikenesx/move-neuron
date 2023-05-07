from collections import defaultdict

from keras.models import load_model

from settings import MODEL_NAME
from utils import process_data_lines

DIFFERENT_MOVES_PATH = 'different_moves.txt'


def main():
    model = load_model(MODEL_NAME)

    with open(DIFFERENT_MOVES_PATH) as file:
        lines = file.readlines()

    processed_lines = process_data_lines(lines=lines)


if __name__ == '__main__':
    main()
