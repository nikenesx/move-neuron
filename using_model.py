import numpy as np

from keras.models import load_model

from charts_utils import draw_plot
from settings import MODEL_NAME, DATA_VECTORS_START_LINE
from utils import process_data_lines

DIFFERENT_MOVES_PATH = 'different_moves_2.txt'


def time_from_line(line):
    splitted_line = line.split(' ')
    return float(splitted_line[0])


def main():
    model = load_model(MODEL_NAME)
    with open('max_value_dataset1.txt') as file:
        max_value = float(file.readline())

    with open(DIFFERENT_MOVES_PATH) as file:
        lines = file.readlines()

    time_values = [time_from_line(line=line) for line in lines[DATA_VECTORS_START_LINE:] if line.strip()]
    processed_lines = process_data_lines(lines=lines)

    time_values = time_values
    processed_lines_float = []
    for line in processed_lines:
        processed_lines_float.append(list(map(lambda x: float(x), line))[:4])

    processed_lines_float = np.array(processed_lines_float) / max_value
    result_values = []
    x_list = []
    for i in range(0, len(processed_lines_float), 100):
        x = np.expand_dims(processed_lines_float[i], axis=0)
        predicted_value = model.predict(x)
        result_values.append(np.argmax(predicted_value))
        x_list.append(time_values[i])

    draw_plot(x_list, result_values)


if __name__ == '__main__':
    main()
