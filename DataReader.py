from collections import defaultdict
from pathlib import Path
from typing import Iterable

from settings import DATASET_PATHS


class DataReader:
    """Класс для считывая набора данных для обучения нейронной сети"""

    __instance = None
    dataset_paths = DATASET_PATHS

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.__instance = super(DataReader, cls).__new__(cls)
        return cls.__instance

    def read_data_by_move_types(self) -> dict[str, list[str]]:
        """
        Получение данных их файлов по типам движений.

        Считывает 'сырые' входные данный со всех директорий, указанных в settings.DATASET_PATHS,
        и формирует из них словарь, где ключ - транслит названия типа движения, а значение - список всех строк,
        найденные для этого типа движения.

        :return: Словарь со строками по типам движений.
        """
        dataset_by_move_types: dict[str, list[str]] = defaultdict(list)

        # Проходим по всем директориям, в которых лежат необработанные данные
        for dataset_path in self.dataset_paths:
            # В директории с данными проходим по всем папкам, отсортированным по типу движения
            for move_type_path in Path(dataset_path).iterdir():
                # Проходим по всем файлам с данными конкретного типа движения
                for data_file in move_type_path.iterdir():
                    with open(data_file) as file:
                        data_lines: list[str] = file.readlines()

                    dataset_by_move_types[move_type_path.stem].extend(data_lines)

        return dataset_by_move_types
