from collections import defaultdict

from pathlib import Path
from typing import Iterable, Iterator

DATASET_PATH = 'dataset2/'
# Строка, с которой начинаются векторы значения во входных файлах с данными
DATA_VECTORS_START_LINE = 7


def read_input_data() -> dict[str, list[list[str]]]:
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

            data_lines: Iterable[str] = lines[DATA_VECTORS_START_LINE:]
            striped_lines: Iterator[str] = map(lambda line: line.strip(), data_lines)
            filtered_lines: Iterator[str] = filter(lambda line: line, striped_lines)
            splitted_lines: Iterator[list[str]] = map(lambda line: line.split()[1:], filtered_lines)

            move_type_data_dict[move_type_dir.stem].extend(splitted_lines)

    return move_type_data_dict
