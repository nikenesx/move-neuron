# Строка, с которой начинаются векторы значения во входных файлах с данными
# (изменять только в случае изменения формата входных данных).
from pathlib import Path

DATA_VECTORS_START_LINE: int = 7


# Количество датчиков, которое используется для обучения сети.
SENSORS_COUNT: int = 4


AVERAGE_READINGS_COUNT: int = 10
MAX_READINGS_COUNT: int = 50

TIME_PER_READING = 0.002

# Директория с входными данными.
DATASET_PATH: str = 'datasets/dataset3'

DATASET_PATHS: list[str] = ['datasets/dataset4']
DIFFERENT_MOVES_PATH: str = 'datasets/different_moves4.txt'

VALUES_THRESHOLDS = {
    'fist': {'sensor': 1, 'threshold': 130},
    'lead': {'sensor': 0, 'threshold': 75},
    'rest': {'sensor': 0, 'threshold': 0},
    'cast': {'sensor': 3, 'threshold': 120},
    'extension': {'sensor': 3, 'threshold': 130},
    'bending': {'sensor': 1, 'threshold': 260},
}


THRESHOLD_FUNCTION_VALUES = {
    'fist': {0: 100, 1: 200, 2: 125, 3: 75},
    'lead': {0: 150, 1: 0, 2: 0, 3: 75},
    'cast': {0: 0, 1: 10, 2: 75, 3: 150},
    'extension': {0: 50, 1: 0, 2: 10, 3: 175},
    'bending': {0: 0, 1: 200, 2: 200, 3: 0},
}


CHARTS_PATH = Path('charts')


class TrainModelOptions:
    """Класс с настройками для обучения сети"""

    # Размер батча. Определяет, через какое кол-во прогона входных векторов будут корректироваться веса.
    # Позволяет быстрее обучаться сети.
    # Возможные значения: от 0 до 10.000
    # Рекомендованные значения: от 10 до 1.000
    BATCH_SIZE: int = 100

    # Количество эпох обучения. Определяет, сколько раз будут прогоняться входные данные. Подача всех векторов на вход
    # для обучения один раз = 1 эпоха.
    # Возможные значения: от 1 до ∞
    EPOCHS_COUNT: int = 30

    # Размер выборки валидации. Определяет, какой процент от входных данных будет использоваться для выборки валидации
    # (необходимо для предотвращения переобучения модели).
    # Возможные значения: от 0.0 до 1.0
    # Рекомендованные значения: от 0.1 до 0.3
    VALIDATION_SPLIT: float = 0.15

    # Размер тестовой выборки
    TESTING_SPLIT: float = 0.05

    # Название обученной нейронной сети (расширение .h5 обязательно)
    MODEL_NAME: str = 'move_types_model.h5'
