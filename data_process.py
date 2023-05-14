import copy
from collections import defaultdict
from itertools import chain
from pathlib import Path

from charts_utils import draw_plot
from constants import MoveTypes
from settings import DATASET_PATHS, SENSORS_COUNT, DATA_VECTORS_START_LINE, AVERAGE_READINGS_COUNT, MAX_READINGS_COUNT, \
    VALUES_THRESHOLDS, CHARTS_PATH


class DataProcess:
    """Класс для считывая набора данных для обучения нейронной сети"""

    __instance = None

    dataset_by_move_types = defaultdict(list)
    normalized_dataset = None

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

    def _convert_data_dict_values_to_lists(self) -> dict[str, list[list[float]]]:
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

    def _calculate_average_window(self, *, dataset_dict: dict[str, list[list[float]]]) -> dict[str, list[list[float]]]:
        """
        Нормализует входной датасет, вычисляя среднее-скользящее каждого элемента
        """
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

    def _calculate_max_window(self, *, dataset_dict: dict[str, list[list[float]]]) -> dict[str, list[list[float]]]:
        """
        Нормализует датасет, находя максимальное значение в окне MAX_READINGS_COUNT каждого датчика
        """
        calculated_dict = defaultdict(list)

        for move_type, lines_lists in dataset_dict.items():
            start_reading = 0
            for sensor_number in range(self.sensors_count):
                current_index = 0
                while True:
                    try:
                        list_to_get_max = [
                            lines_lists[line_number][sensor_number]
                            for line_number in range(start_reading, start_reading + MAX_READINGS_COUNT)
                        ]
                        max_window_value = max(list_to_get_max)
                    except IndexError:
                        break

                    try:
                        calculated_dict[move_type][current_index].append(max_window_value)
                    except IndexError:
                        calculated_dict[move_type].append([])
                        calculated_dict[move_type][current_index].append(max_window_value)

                    current_index += 1
                    start_reading += MAX_READINGS_COUNT

                start_reading = 0

        return calculated_dict

    def get_normalized_dataset(self) -> dict[str, list[list[float]]]:
        """Считывает набор данных из файлов, нормализует и возвращает его"""
        converted_dataset = self._convert_data_dict_values_to_lists()
        average_calculated_dataset = self._calculate_average_window(dataset_dict=converted_dataset)
        max_calculated_dataset = self._calculate_max_window(dataset_dict=average_calculated_dataset)
        normalized_by_threshold, source_data = self.normalize_dataset_by_threshold(
            dataset=average_calculated_dataset,
            source_dataset=converted_dataset,
        )

        self.save_charts_normalized_data(
            converted_dataset=converted_dataset,
            average_calculated_dataset=average_calculated_dataset,
            max_calculated_dataset=max_calculated_dataset,
        )
        self.save_charts_source_data(source_data=source_data)

        return normalized_by_threshold

    @staticmethod
    def normalize_dataset_by_threshold(*, dataset: dict, source_dataset: dict):
        calculated_dict = defaultdict(list)
        source_dataset_dict = defaultdict(list)

        for move_type, lines in dataset.items():
            sensor_number = VALUES_THRESHOLDS[move_type]['sensor']
            threshold = VALUES_THRESHOLDS[move_type]['threshold']

            for index in range(len(lines)):
                if lines[index][sensor_number] >= threshold:
                    calculated_dict[move_type].append(lines[index])
                    source_dataset_dict[move_type].append(source_dataset[move_type][index])

        return calculated_dict, source_dataset_dict

    def save_charts_normalized_data(self, *, converted_dataset, average_calculated_dataset, max_calculated_dataset):
        for move_type in max_calculated_dataset:
            for sensor in range(self.sensors_count):
                y1 = tuple(values[sensor] for values in converted_dataset[move_type])[:5000]
                x1 = range(len(y1))

                y2 = tuple(values[sensor] for values in average_calculated_dataset[move_type])[:5000]
                x2 = range(len(y2))

                y3 = tuple([values[sensor]] * MAX_READINGS_COUNT for values in max_calculated_dataset[move_type])
                y3_ext = list(chain(*y3))[:5000]
                x3 = range(len(y3_ext))

                move_type_translit = MoveTypes.TRANSLATIONS[move_type]
                plot_name = Path(
                    f'{move_type}_normalized_{sensor + 1}_sensor'
                )

                chart_title = f'Исходные и нормализованные значения | {move_type_translit} | {sensor + 1} датчик'
                label1 = 'Исходное значение'
                label2 = 'Скользящее среднее'
                label3 = 'Максимальное значение в окрестности'

                draw_plot(
                    x1=x1, y1=y1, label1=label1,
                    x2=x2, y2=y2, label2=label2,
                    x3=x3, y3=y3_ext, label3=label3,
                    path_to_save=CHARTS_PATH / plot_name,
                    chart_title=chart_title,
                )

    def save_charts_source_data(self, *, source_data):
        for move_type, lines in source_data.items():
            for sensor in range(self.sensors_count):
                y = tuple(line[sensor] for line in lines)
                x = range(len(y))

                move_type_translit = MoveTypes.TRANSLATIONS[move_type]

                path_to_save = CHARTS_PATH / Path('normalized_by_threshold')
                path_to_save.mkdir(parents=True, exist_ok=True)
                plot_name = Path(
                    f'{move_type}_source_dataset_{sensor + 1}_sensor'
                )

                chart_title = f'Исходные активные значения | {move_type_translit} | {sensor + 1} датчик'
                label = 'Исходное значение'

                draw_plot(x1=x, y1=y, path_to_save=path_to_save / plot_name, chart_title=chart_title, label1=label)


if __name__ == '__main__':
    data_process = DataProcess()
    data_process.get_normalized_dataset()
