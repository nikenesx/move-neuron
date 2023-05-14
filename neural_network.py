import os

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from charts_utils import save_loss_chart, save_accuracy_chart
from constants import MoveTypes
from data_process import DataProcess
from settings import TrainModelOptions
from utils import read_input_data, get_input_and_expected_vectors, get_max_sensor_value

# Отключение использования CUDA ядер
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def train_model(saved_model_name: str, show_stat: bool = False) -> None:
    """
    Получает набор входных данных и обучает модель для предсказания движения человеческой кисти.

    :param saved_model_name: Название обученной модели
    :param show_stat: Если True, то сохраняются графики точности и потерь
    """
    data_process = DataProcess()
    sensor_data_vectors = data_process.get_normalized_dataset()
    input_vectors, result_values = get_input_and_expected_vectors(input_data=sensor_data_vectors)
    max_value = get_max_sensor_value(data=input_vectors)

    with open(f'max_value.txt', 'w') as file:
        file.write(str(max_value))

    input_vectors = np.array(input_vectors) / max_value
    result_values_cat = to_categorical(np.array(result_values), len(MoveTypes.NUMS))

    input_vectors, result_values_cat = unison_shuffled_copies(input_vectors, result_values_cat)

    test_percent = 0.1
    test_len = int(len(input_vectors) * test_percent)
    input_vectors_test = input_vectors[:test_len]
    result_values_cat_test = result_values_cat[:test_len]

    input_vectors = input_vectors[test_len:]
    result_values_cat = result_values_cat[test_len:]

    model = Sequential([
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(6, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=['accuracy'])

    history = model.fit(
        x=input_vectors,
        y=result_values_cat,
        epochs=TrainModelOptions.EPOCHS_COUNT,
        batch_size=TrainModelOptions.BATCH_SIZE,
        validation_split=TrainModelOptions.VALIDATION_SPLIT,
    )
    model.save(saved_model_name)

    accuracy = model.evaluate(input_vectors_test, result_values_cat_test)
    accuracy_value = round(float(accuracy[1] * 100), 2)
    print(f'Точность: {accuracy_value}%')

    if show_stat:
        save_loss_chart(fitted_model=model, model_history=history, batch_size=TrainModelOptions.BATCH_SIZE)
        save_accuracy_chart(fitted_model=model, model_history=history, batch_size=TrainModelOptions.BATCH_SIZE)


if __name__ == '__main__':
    train_model(saved_model_name=TrainModelOptions.MODEL_NAME, show_stat=False)
