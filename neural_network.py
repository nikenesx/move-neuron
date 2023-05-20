import os
from pathlib import Path

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical

from constants import MoveTypes
from data_process import DataProcess
from settings import TrainModelOptions, SENSORS_COUNT

# Отключение использования CUDA ядер
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Network:

    MODEL_NAME = TrainModelOptions.MODEL_NAME
    TRAINED_MODEL = None

    def __init__(self):
        if Path(self.MODEL_NAME).exists():
            self.TRAINED_MODEL = load_model(self.MODEL_NAME)

    def train_model(self) -> None:
        if self.TRAINED_MODEL:
            print(f'Найдена обученная модель {self.MODEL_NAME}. Обучить новую?')
            answer = input('Да/нет? : ')
            if answer.strip().lower() != 'да':
                return

        data_process = DataProcess()
        sensor_data_vectors = data_process.get_normalized_dataset_to_fit_model()

        input_vectors, result_values = self.get_input_and_expected_vectors(input_data=sensor_data_vectors)

        input_vectors = np.array(input_vectors)
        result_values_cat = to_categorical(np.array(result_values), len(MoveTypes.NUMS))

        input_vectors, result_values_cat = self.unison_shuffled_copies(input_vectors, result_values_cat)

        test_len = int(len(input_vectors) * TrainModelOptions.TESTING_SPLIT)

        input_vectors_test, result_values_cat_test = input_vectors[:test_len], result_values_cat[:test_len]
        input_vectors, result_values_cat = input_vectors[test_len:], result_values_cat[test_len:]

        model = Sequential([
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(6, activation='softmax')
        ])

        model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=['accuracy'])

        model.fit(
            x=input_vectors,
            y=result_values_cat,
            epochs=TrainModelOptions.EPOCHS_COUNT,
            batch_size=TrainModelOptions.BATCH_SIZE,
            validation_split=TrainModelOptions.VALIDATION_SPLIT,
        )
        model.save(self.MODEL_NAME)

        accuracy = model.evaluate(input_vectors_test, result_values_cat_test)
        accuracy_value = round(float(accuracy[1] * 100), 2)
        print(f'Точность: {accuracy_value}%')

    @staticmethod
    def unison_shuffled_copies(first_collection, second_collection):
        """Перемешивает значение двух коллекций "в унисон"""
        assert len(first_collection) == len(second_collection)
        shuffled_indexes = np.random.permutation(len(first_collection))

        return first_collection[shuffled_indexes], second_collection[shuffled_indexes]

    @staticmethod
    def get_input_and_expected_vectors(*, input_data) -> tuple[list[list[float]], list[int]]:
        """
        Обрабатывает словарь с входными данными, возвращая преобразованные значения в числа
        и список с ожидаемыми значениями.
        """
        input_vectors = []
        result_values = []

        for key, lines in input_data.items():
            for line in lines:
                digit_lines = list(map(lambda value: float(value), line))
                input_vectors.append(digit_lines[:SENSORS_COUNT])
                result_values.append(MoveTypes.NUMS[key])

        return input_vectors, result_values

    @staticmethod
    def get_max_sensor_value(*, data: list[list[float]]) -> float:
        """Возвращает максимальное значение из входных векторов"""
        max_value = 0
        for vector in data:
            lst_max = max(vector)
            if lst_max > max_value:
                max_value = lst_max

        return max_value


if __name__ == '__main__':
    network = Network()
    network.train_model()
