from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from settings import DATASET_PATH, DATA_VECTORS_START_LINE, SENSORS_COUNT, WINDOW_SIZE


def normalize_input_data(window_size: int = WINDOW_SIZE):
    dataset_path: Path = Path(DATASET_PATH)
    move_type_data_dict: dict = defaultdict(list)

    for move_type_dir in dataset_path.iterdir():
        for data_file in move_type_dir.iterdir():
            with open(data_file) as file:
                lines: Iterable[str] = file.readlines()

            data_lines: Iterable[str] = lines[DATA_VECTORS_START_LINE:]
            striped_lines: Iterator[str] = map(lambda line: line.strip(), data_lines)
            filtered_lines: Iterator[str] = filter(lambda line: line, striped_lines)

            processed_lines = list(map(lambda line: line.split(), filtered_lines))
            move_type_data_dict[move_type_dir.stem].extend(processed_lines)

    result_data = defaultdict(list)

    for key, lines in move_type_data_dict.items():
        for line in lines:
            digit_lines = list(map(lambda value: float(value), line))
            result_data[key].append(digit_lines[:SENSORS_COUNT + 1])

    normalized_dict = defaultdict(list)
    temp_list = []

    if window_size <= 0:
        print(f'Окно: {window_size}')
        for key, value in result_data.items():
            print(f'Тип движения: {key} | Количество векторов: {len(value)}')
        return result_data


    for move_type, lines in result_data.items():
        start_time = window_size
        for line in lines:
            if line[0] <= start_time:
                temp_list.append(line[1:])
            else:
                normalized_dict[move_type].append(temp_list[0])
                for i in range(SENSORS_COUNT):
                    for tmp_lst in temp_list[1:]:
                        normalized_dict[move_type][-1][i] = max(normalized_dict[move_type][-1][i], tmp_lst[i])

                temp_list = []
                start_time += window_size

    print(f'Окно: {window_size}')
    for key, value in normalized_dict.items():
        print(f'Тип движения: {key} | Количество векторов: {len(value)}')

    return normalized_dict


if __name__ == '__main__':
    normalize_input_data()
