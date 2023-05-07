import os

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from constants import MoveTypes
from data_utils import read_input_data

# Отключение использования CUDA ядер
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

LOSS_CHARTS_DIR_NAME = 'loss_charts/'
ACCURACY_CHARTS_DIR_NAME = 'accuracy_charts/'
DIABETES_SAVED_MODEL = 'diabetes_model.h5'


def train_diabetes_name(saved_model_name: str, show_stat: bool = False) -> None:
    """
    Получает датасет обучает модель для предсказания развития диабета.
    :param saved_model_name: название обученной модели
    :param show_stat: сохраняет графики точности и потерь, если True
    """
    input_vectors = []
    result_values = []

    move_data = read_input_data()
    for key, lines in move_data.items():
        for line in lines:
            digit_lines = list(map(lambda x: round(float(x), 5), line))
            input_vectors.append(digit_lines[:8])
            result_values.append(MoveTypes.NUMS[key])

    max_value = 0
    for lst in input_vectors:
        lst_max = max(lst)
        if lst_max > max_value:
            max_value = lst_max


    input_vectors = np.array(input_vectors) / max_value
    result_values = np.array(result_values)
    result_values_cat = to_categorical(result_values, 6)

    model = Sequential([
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(6, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=['accuracy'])

    batch_size = 100
    history = model.fit(input_vectors, result_values_cat, epochs=50, batch_size=batch_size, validation_split=0.3)
    cast_test = np.array([1648.38788, 1651.61211, 1650.80605, 1655.64240, 1664.50903, 1637.90913, 1653.22423, 1641.13336]) / max_value
    fist_test = np.array([1670.15144, 1662.09086, 1655.64240, 1655.64240, 1654.03028, 1645.96971, 1649.99999, 1649.99999]) / max_value
    rest_test = np.array([1654.83634, 1652.41817, 1644.35759, 1654.03028, 1662.09086, 1632.26673, 1654.83634, 1642.74548]) / max_value

    cast_test = np.expand_dims(cast_test, axis=0)
    fist_test = np.expand_dims(fist_test, axis=0)
    rest_test = np.expand_dims(rest_test, axis=0)

    res_cast = model.predict(cast_test)
    print(res_cast)
    print(np.argmax(res_cast))

    res_fist = model.predict(fist_test)
    print(res_fist)
    print(np.argmax(res_fist))

    res_rest = model.predict(rest_test)
    print(res_rest)
    print(np.argmax(res_rest))

    # model.save(saved_model_name)
    #
    # if show_stat:
    #     save_loss_chart(fitted_model=model, model_history=history, batch_size=batch_size)
    #     save_accuracy_chart(fitted_model=model, model_history=history, batch_size=batch_size)
    #
    #     accuracy = model.evaluate(input_vectors, result_values)
    #     accuracy_value = round(float(accuracy[1] * 100), 2)
    #     print(f'Точность: {accuracy_value}%')


if __name__ == '__main__':
    model_name = DIABETES_SAVED_MODEL
    train_diabetes_name(saved_model_name=model_name, show_stat=True)
