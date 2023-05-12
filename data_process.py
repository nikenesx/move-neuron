import copy
from collections import defaultdict
from pathlib import Path

from settings import DATASET_PATHS, SENSORS_COUNT, DATA_VECTORS_START_LINE, AVERAGE_READINGS_COUNT


class DataProcess:
    """Класс для считывая набора данных для обучения нейронной сети"""

    __instance = None

    dataset_by_move_types = defaultdict(list)
    dataset_paths = DATASET_PATHS
    sensors_count = SENSORS_COUNT

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.__instance = super(DataProcess, cls).__new__(cls)
        return cls.__instance

    def _read_data_by_move_types(self) -> dict[str, list[str]]:
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

                    striped_lines = map(lambda line: line.strip(), data_lines)
                    not_empty_lines = filter(lambda line: line, striped_lines)
                    dataset_by_move_types[move_type_path.stem].extend(list(not_empty_lines))

        return dataset_by_move_types

    def _convert_data_dict_values_to_lists(self):
        """
        Проходит по каждому типу движения, преобразовывая каждую строку с данными в список с численными значениями.
        """
        if self.dataset_by_move_types:
            return self.dataset_by_move_types

        dataset_from_files = self._read_data_by_move_types()

        for move_type, lines in dataset_from_files.items():
            splited_lines = list(map(lambda line: line.split(' ')[1:self.sensors_count + 1], lines))
            lines_float_values = list(map(lambda line: list(map(lambda value: float(value), line)), splited_lines))

            self.dataset_by_move_types[move_type].extend(list(lines_float_values))

        return self.dataset_by_move_types

    def _calculate_average_window(self, *, dataset_dict):
        calculated_dict = copy.deepcopy(dataset_dict)

        for move_type, lines_lists in calculated_dict.items():
            start_reading = 0
            sum_values = 0

            for sensor_number in range(self.sensors_count):
                while True:

                    try:
                        for line_number in range(start_reading, start_reading + AVERAGE_READINGS_COUNT):
                            sum_values += lines_lists[line_number][sensor_number]
                    except IndexError:
                        for i in range(start_reading, line_number):
                            index = abs(line_number - start_reading)
                            lines_lists[i][sensor_number] = abs(lines_lists[i][sensor_number] - sum_values / index)

                        break

                    readings_average_value = sum_values / AVERAGE_READINGS_COUNT

                    for line_number in range(start_reading, start_reading + AVERAGE_READINGS_COUNT):
                        lines_lists[line_number][sensor_number] = abs(
                            lines_lists[line_number][sensor_number] - readings_average_value
                        )

                    sum_values = 0
                    start_reading += AVERAGE_READINGS_COUNT

                start_reading = 0
                sum_values = 0

        return calculated_dict

    def normalize_dataset_by_move_types(self):
        converted_dataset = self._convert_data_dict_values_to_lists()
        average_calculated_dataset = self._calculate_average_window(dataset_dict=converted_dataset)
        print(average_calculated_dataset)


if __name__ == '__main__':
    data_process = DataProcess()
    data_process.normalize_dataset_by_move_types()
