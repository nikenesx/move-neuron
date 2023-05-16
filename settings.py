# Строка, с которой начинаются векторы значения во входных файлах с данными
# (изменять только в случае изменения формата входных данных).
from pathlib import Path

DATA_VECTORS_START_LINE: int = 7


# Количество датчиков, которое используется для обучения сети.
SENSORS_COUNT: int = 4


AVERAGE_READINGS_COUNT: int = 15
MAX_READINGS_COUNT: int = 50

TIME_PER_READING = 0.002

# Директория с входными данными.
DATASET_PATH: str = 'datasets/dataset3'

DATASET_PATHS: list[str] = ['datasets/dataset3']  #, 'datasets/dataset2', 'datasets/dataset1']

VALUES_THRESHOLDS = {
    'fist': {'sensor': 1, 'threshold': 120},
    'lead': {'sensor': 0, 'threshold': 65},
    'rest': {'sensor': 0, 'threshold': 0},
    'cast': {'sensor': 3, 'threshold': 110},
    'extension': {'sensor': 3, 'threshold': 120},
    'bending': {'sensor': 1, 'threshold': 250},
}

CHARTS_PATH = Path('charts')


class TrainModelOptions:
    """Класс с настройками для обучения сети"""

    # Размер батча. Определяет, через какое кол-во прогона входных векторов будут корректироваться веса.
    # Позволяет быстрее обучаться сети.
    # Возможные значения: от 0 до 10.000
    # Рекомендованные значения: от 10 до 1.000
    BATCH_SIZE: int = 32

    # Количество эпох обучения. Определяет, сколько раз будут прогоняться входные данные. Подача всех векторов на вход
    # для обучения один раз = 1 эпоха.
    # Возможные значения: от 1 до ∞
    EPOCHS_COUNT: int = 10

    # Размер выборки валидации. Определяет, какой процент от входных данных будет использоваться для выборки валидации
    # (необходимо для предотвращения переобучения модели).
    # Возможные значения: от 0.0 до 1.0
    # Рекомендованные значения: от 0.1 до 0.3
    VALIDATION_SPLIT: float = 0.10

    # Название обученной нейронной сети (расширение .h5 обязательно)
    MODEL_NAME: str = 'move_types_model.h5'
