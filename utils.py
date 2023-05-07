from collections import defaultdict

from pathlib import Path
from typing import Iterable, Iterator

from constants import MoveTypes
from settings import DATASET_PATH, DATA_VECTORS_START_LINE, SENSORS_COUNT

INPUT_VECTOR_TYPE = dict[str, list[list[str]]]
PROCESSED_VECTOR_TYPE = dict[str, list[list[float]]]


def read_input_data() -> INPUT_VECTOR_TYPE:
    """
    Считывает показания датчиков с файлов и формирует словарь,
    где ключ - транслит типа движения, а значение - список со списком значения датчиков.
    """
    dataset_path: Path = Path(DATASET_PATH)
    move_type_data_dict: dict = defaultdict(list)

    for move_type_dir in dataset_path.iterdir():
        for data_file in move_type_dir.iterdir():
            with open(data_file) as file:
                lines: Iterable[str] = file.readlines()

            processed_lines = process_data_lines(lines=lines)
            move_type_data_dict[move_type_dir.stem].extend(processed_lines)

    return move_type_data_dict


def process_data_lines(lines: Iterable[str]) -> Iterator[list[str]]:
    data_lines: Iterable[str] = lines[DATA_VECTORS_START_LINE:]
    striped_lines: Iterator[str] = map(lambda line: line.strip(), data_lines)
    filtered_lines: Iterator[str] = filter(lambda line: line, striped_lines)

    return map(lambda line: line.split()[1:], filtered_lines)


def get_input_and_expected_vectors(*, input_data: INPUT_VECTOR_TYPE) -> tuple[list[list[float]], list[int]]:
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


def get_max_sensor_value(*, data: list[list[float]]) -> float:
    """Возвращает максимальное значение из входных векторов"""
    max_value = 0
    for vector in data:
        lst_max = max(vector)
        if lst_max > max_value:
            max_value = lst_max

    return max_value
