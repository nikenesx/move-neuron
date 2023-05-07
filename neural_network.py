import os

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from charts_utils import save_loss_chart, save_accuracy_chart
from constants import MoveTypes
from settings import TRAIN_MODEL_OPTIONS, MODEL_NAME, DATASET_PATH
from utils import read_input_data, get_input_and_expected_vectors, get_max_sensor_value

# Отключение использования CUDA ядер
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_model(saved_model_name: str, show_stat: bool = False) -> None:
    """
    Получает набор входных данных и обучает модель для предсказания движения человеческой кисти.

    :param saved_model_name: Название обученной модели
    :param show_stat: Если True, то сохраняются графики точности и потерь
    """
    sensor_data_vectors = read_input_data()
    input_vectors, result_values = get_input_and_expected_vectors(input_data=sensor_data_vectors)
    max_value = get_max_sensor_value(data=input_vectors)

    input_vectors = np.array(input_vectors) / max_value
    result_values_cat = to_categorical(np.array(result_values), len(MoveTypes.NUMS))

    model = Sequential([
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(6, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=['accuracy'])

    history = model.fit(
        x=input_vectors,
        y=result_values_cat,
        epochs=TRAIN_MODEL_OPTIONS['epochs_count'],
        batch_size=TRAIN_MODEL_OPTIONS['batch_size'],
        validation_split=TRAIN_MODEL_OPTIONS['validation_split'],
    )
    model.save(saved_model_name)
    with open(f'max_value_{DATASET_PATH}.txt', 'w') as file:
        file.write(str(max_value))

    accuracy = model.evaluate(input_vectors, result_values_cat)
    accuracy_value = round(float(accuracy[1] * 100), 2)
    print(f'Точность: {accuracy_value}%')

    if show_stat:
        save_loss_chart(fitted_model=model, model_history=history, batch_size=TRAIN_MODEL_OPTIONS['batch_size'])
        save_accuracy_chart(fitted_model=model, model_history=history, batch_size=TRAIN_MODEL_OPTIONS['batch_size'])


if __name__ == '__main__':
    train_model(saved_model_name=MODEL_NAME, show_stat=False)
