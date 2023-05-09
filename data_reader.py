from collections import defaultdict
from pathlib import Path

from settings import DATASET_PATHS, SENSORS_COUNT, DATA_VECTORS_START_LINE


class DataReader:
    """Класс для считывая набора данных для обучения нейронной сети"""

    __instance = None

    dataset_by_move_types = None
    dataset_paths = DATASET_PATHS
    sensors_count = SENSORS_COUNT

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.__instance = super(DataReader, cls).__new__(cls)
        return cls.__instance

    def _read_data_by_move_types(self) -> None:
        """
        Получение данных их файлов по типам движений.

        Считывает 'сырые' входные данный со всех директорий, указанных в settings.DATASET_PATHS,
        и формирует из них словарь, где ключ - транслит названия типа движения, а значение - список всех строк,
        найденные для этого типа движения.
        Во время считывания отбрасываются пустые строки.
        """
        dataset_by_move_types: dict[str, list[str]] = defaultdict(list)

        # Проходим по всем директориям, в которых лежат необработанные данные
        for dataset_path in self.dataset_paths:
            # В директории с данными проходим по всем папкам, отсортированным по типу движения
            for move_type_path in Path(dataset_path).iterdir():
                # Проходим по всем файлам с данными конкретного типа движения
                for data_file in move_type_path.iterdir():
                    with open(data_file) as file:
                        data_lines: list[str] = file.readlines()[DATA_VECTORS_START_LINE:]

                    striped_lines = list(map(lambda line: line.strip(), data_lines))
                    not_empty_lines = list(filter(lambda line: line, striped_lines))
                    dataset_by_move_types[move_type_path.stem].extend(list(not_empty_lines))

        self.dataset_by_move_types = dataset_by_move_types

    def _convert_data_dict_values_to_lists(self):
        """
        Проходит по каждому типу движения, преобразовывая каждую строку с данными в список с численными значениями.
        """
        if not self.dataset_by_move_types:
            self._read_data_by_move_types()

        for move_type, lines in self.dataset_by_move_types.items():
            splited_lines = map(lambda line: line.split(' ')[:self.sensors_count + 1], lines)
            lines_float_values = map(lambda line: list(map(lambda value: float(value), line)), splited_lines)

            self.dataset_by_move_types[move_type] = list(lines_float_values)
