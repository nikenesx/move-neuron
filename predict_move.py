from pathlib import Path
from typing import Optional

import numpy as np

from charts_utils import draw_plot
from constants import MoveTypes
from data_process import DataProcess
from neural_network import Network
from settings import THRESHOLD_FUNCTION_VALUES, ProcessingType, TIME_PER_READING, CHARTS_PATH


class PredictMove:

    def __init__(self):
        self.data_process = DataProcess()
        self.network = Network()

    @staticmethod
    def _is_overcome_thresholds(*, vector: list[float], thresholds: dict[int, float]) -> bool:
        for index in range(len(vector)):
            if vector[index] < thresholds[index]:
                return False

        return True

    def predict_by_threshold(self, *, vector: list[float]) -> int:
        """
        Определить тип движения через пороговую функцию
        """
        for move, thresholds in THRESHOLD_FUNCTION_VALUES.items():
            if self._is_overcome_thresholds(vector=vector, thresholds=thresholds):
                return MoveTypes.NUMS[move]

        return MoveTypes.NUMS[MoveTypes.REST]

    def predict_by_network(self, *, vector: list[float]) -> int:
        """
        Определить тип движения через нейронную сеть
        """
        vector = np.expand_dims(vector, axis=0)
        return self.network.TRAINED_MODEL.predict(vector, verbose=0)

    def process_different_moves_file(
        self,
        *,
        processing: str,
        values_count: Optional[int] = None,
        is_draw_plot: bool = True,
    ) -> None:
        """
        Обработать файл с разными движениями
        """
        dataset = self.data_process.get_normalized_different_moves_dataset()
        if values_count:
            dataset = dataset[:values_count]

        if processing == ProcessingType.NETWORK:
            process_function = self.predict_by_network
        elif processing == ProcessingType.FUNCTION:
            process_function = self.predict_by_threshold
        else:
            raise ValueError(f'Unknown processing method. Use {ProcessingType.METHODS}')

        results = np.array([])
        [np.append(results, process_function(vector=vector)) for vector in dataset]

        if is_draw_plot:
            self.draw_different_moves_chart(results=results, processing=processing)

    @staticmethod
    def draw_different_moves_chart(*, results: np.array, processing: str):
        """
        Сохранить график с результатами
        """
        x = list(map(lambda val: val * TIME_PER_READING, results))
        chart_name = f'results_{processing}'

        path_to_save = CHARTS_PATH / Path(f'results/{chart_name}')
        path_to_save.mkdir(parents=True, exist_ok=True)

        draw_plot(
            x1=x, y1=results, label1='Результат',
            path_to_save=CHARTS_PATH,
            chart_title='Результаты',
        )


if __name__ == '__main__':
    pr = PredictMove()
    pr.process_different_moves_file(processing=ProcessingType.FUNCTION, values_count=5000)
