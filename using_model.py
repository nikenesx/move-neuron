import numpy as np

from keras.models import load_model

from charts_utils import draw_plot
from settings import MODEL_NAME, DATA_VECTORS_START_LINE
from utils import process_data_lines, process_data_lines2

DIFFERENT_MOVES_PATH = 'datasets/different_moves_2.txt'


def time_from_line(line):
    splitted_line = line.split(' ')
    return float(splitted_line[0])


def draw_azaza():
    with open('max_value_dataset2.txt') as file:
        max_value = float(file.readline())

    with open(DIFFERENT_MOVES_PATH) as file:
        lines = file.readlines()

    time_values = [time_from_line(line=line) for line in lines[DATA_VECTORS_START_LINE:] if line.strip()]
    processed_lines = process_data_lines2(lines=lines)
    x1 = [float(line[1]) for line in processed_lines]
    x2 = [float(line[2]) for line in processed_lines]
    x3 = [float(line[3]) for line in processed_lines]
    x4 = [float(line[4]) for line in processed_lines]
    draw_plot(time_values, x1, name='plot1', label='1 Датчик')
    draw_plot(time_values, x2, name='plot2', label='2 Датчик')
    draw_plot(time_values, x3, name='plot3', label='3 Датчик')
    draw_plot(time_values, x4, name='plot4', label='4 Датчик')


def main():
    model = load_model(MODEL_NAME)
    with open('max_value_dataset2.txt') as file:
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
    # main()
    draw_azaza()
