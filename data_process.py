import copy
import shutil
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Optional

from charts_utils import draw_plot
from constants import MoveTypes
from settings import DATASET_PATHS, SENSORS_COUNT, DATA_VECTORS_START_LINE, AVERAGE_READINGS_COUNT, MAX_READINGS_COUNT, \
    VALUES_THRESHOLDS, CHARTS_PATH, TIME_PER_READING

PATH_UN = Path(CHARTS_PATH, 'uniq')


class DataProcess:
    """Класс для считывая набора данных для обучения нейронной сети"""

    __instance = None

    dataset_by_move_types = defaultdict(list)
    normalized_dataset = None

    dataset_paths = DATASET_PATHS
    sensors_count = SENSORS_COUNT

    def __new__(cls, moves_chart: tuple[str], sensors_chart: tuple[int], segments_chart: tuple[list[str]]):
        if not hasattr(cls, 'instance'):
            cls.__instance = super(DataProcess, cls).__new__(cls)
        return cls.__instance

    def __init__(
        self,
        moves_chart: Optional[tuple[str]] = None,
        sensors_chart: Optional[tuple[int]] = None,
        segments_chart: Optional[tuple[list[str]]] = None,
    ):
        self.moves_chart = moves_chart
        self.sensors_chart = sensors_chart
        self.segments_chart = segments_chart

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
        moves_list = self.moves_chart or max_calculated_dataset.keys()
        sensors_list = self.sensors_chart or range(self.sensors_count)

        if self.moves_chart or self.sensors_chart or self.segments_chart:
            path_to_save = PATH_UN
        else:
            path_to_save = CHARTS_PATH

        path_to_save.mkdir(exist_ok=True, parents=True)

        for move_type in moves_list:
            for sensor in sensors_list:
                y1 = tuple(values[sensor] for values in converted_dataset[move_type])
                x1 = range(len(y1))

                y2 = tuple(values[sensor] for values in average_calculated_dataset[move_type])
                x2 = range(len(y2))

                y3 = tuple([values[sensor]] * MAX_READINGS_COUNT for values in max_calculated_dataset[move_type])
                y3_ext = list(chain(*y3))
                x3 = range(len(y3_ext))

                move_type_translit = MoveTypes.TRANSLATIONS[move_type]
                plot_name = Path(
                    f'{move_type}_normalized_{sensor + 1}_sensor'
                )

                chart_title = f'Исходные и нормализованные значения | {move_type_translit} | {sensor + 1} канал'
                label1 = 'Исходное значение'
                label2 = 'Скользящее среднее'
                label3 = 'Максимальное значение в окрестности'

                x1_t = list(map(lambda x: x * TIME_PER_READING, x1))
                x2_t = list(map(lambda x: x * TIME_PER_READING, x2))
                x3_t = list(map(lambda x: x * TIME_PER_READING, x3))
                if not self.segments_chart:
                    max_values = 5000
                    draw_plot(
                        x1=x1_t[:max_values], y1=y1[:max_values], label1=label1,
                        x2=x2_t[:max_values], y2=y2[:max_values], label2=label2,
                        # x3=x3_t[:max_values], y3=y3_ext[:max_values], label3=label3,
                        path_to_save=path_to_save / plot_name,
                        chart_title=chart_title,
                    )
                    continue

                x1_dict = {}
                x2_dict = {}
                x3_dict = {}

                for i in x1:
                    x1_dict[i * TIME_PER_READING] = y1[i]
                for i in x2:
                    x2_dict[i * TIME_PER_READING] = y2[i]
                for i in x3:
                    x3_dict[i * TIME_PER_READING] = y3_ext[i]

                for segment in self.segments_chart:
                    start = int(segment[0])
                    end = int(segment[1])

                    x1_times_list = list(filter(lambda x: start <= x <= end, x1_dict))
                    x2_times_list = list(filter(lambda x: start <= x <= end, x2_dict))
                    x3_times_list = list(filter(lambda x: start <= x <= end, x3_dict))

                    y1_values = [x1_dict[value] for value in x1_times_list]
                    y2_values = [x2_dict[value] for value in x2_times_list]
                    y3_values = [x3_dict[value] for value in x3_times_list]

                    chart_titles = f'{chart_title} | интервал {start}:{end}'
                    plot_names = f'{str(plot_name)}_{start}_{end}'

                    draw_plot(
                        x1=x1_times_list, y1=y1_values, label1=label1,
                        x2=x2_times_list, y2=y2_values, label2=label2,
                        x3=x3_times_list, y3=y3_values, label3=label3,
                        path_to_save=path_to_save / Path(plot_names),
                        chart_title=chart_titles,
                    )

    def save_charts_source_data(self, *, source_data):
        moves_list = self.moves_chart or source_data.keys()
        sensors_list = self.sensors_chart or range(self.sensors_count)

        if self.moves_chart or self.sensors_chart or self.segments_chart:
            path_to_save = PATH_UN / Path('active_values')
        else:
            path_to_save = CHARTS_PATH / Path('normalized_by_threshold')

        path_to_save.mkdir(exist_ok=True, parents=True)

        for move_type in moves_list:
            for sensor in range(self.sensors_count):
                lines = source_data[move_type]
                y = tuple(line[sensor] for line in lines)
                x = range(len(y))

                move_type_translit = MoveTypes.TRANSLATIONS[move_type]

                plot_name = Path(
                    f'{move_type}_source_dataset_{sensor + 1}_sensor'
                )

                chart_title = f'Исходные активные значения | {move_type_translit} | {sensor + 1} датчик'
                label = 'Исходное значение'
                x = list(map(lambda el: el * TIME_PER_READING, x))

                draw_plot(x1=x, y1=y, path_to_save=path_to_save / plot_name, chart_title=chart_title, label1=label)


def parse_arguments(arguments_list: list[str]):
    moves = None
    sensors = None
    segments = None

    if '--moves' in arguments_list:
        moves_str = arguments_list[arguments_list.index('--moves') + 1]
        moves_split = map(lambda el: el.strip(), moves_str.split(','))
        moves = tuple(filter(lambda el: el in MoveTypes.ALL_TYPES, moves_split))

    if '--sensors' in arguments_list:
        sensors_str = arguments_list[arguments_list.index('--sensors') + 1]
        sensors_split = map(lambda el: int(el.strip()) - 1, sensors_str.split(','))
        sensors = tuple(filter(lambda el: el in range(SENSORS_COUNT), sensors_split))

    if '--segments' in arguments_list:
        segments_str = arguments_list[arguments_list.index('--segments') + 1]
        segments = tuple(map(lambda el: el.strip().split(':'), segments_str.split(',')))

    PATH_UN.mkdir(parents=True, exist_ok=True)

    if moves or sensors or segments:
        shutil.rmtree(PATH_UN)

    PATH_UN.mkdir(parents=True, exist_ok=True)

    return moves, sensors, segments


if __name__ == '__main__':
    moves, sensors, segments = parse_arguments(sys.argv)
    data_process = DataProcess(moves_chart=moves, sensors_chart=sensors, segments_chart=segments)
    data_process.get_normalized_dataset()
