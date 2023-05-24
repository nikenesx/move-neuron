from itertools import chain
from pathlib import Path
from typing import Optional

import numpy as np

from charts_utils import draw_plot
from constants import MoveTypes
from data_process import DataProcess
from neural_network import Network
from settings import THRESHOLD_FUNCTION_VALUES, ProcessingType, TIME_PER_READING, CHARTS_PATH, MAX_READINGS_COUNT


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
        predicted_value = self.network.TRAINED_MODEL.predict(vector, verbose=0)
        return int(np.argmax(predicted_value))

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
        dataset = self.data_process.get_normalized_different_moves_dataset()[:5000]
        if values_count:
            dataset = dataset[:values_count]

        if processing == ProcessingType.NETWORK:
            process_function = self.predict_by_network
        elif processing == ProcessingType.FUNCTION:
            process_function = self.predict_by_threshold
        else:
            raise ValueError(f'Unknown processing method. Use {ProcessingType.METHODS}')

        results = []
        for vector in dataset:
            results.append(process_function(vector=vector))

        if is_draw_plot:
            self.draw_different_moves_chart(results=results, processing=processing)

    @staticmethod
    def draw_different_moves_chart(*, results: np.array, processing: str):
        """
        Сохранить график с результатами
        """
        # results = [[res] * MAX_READINGS_COUNT for res in results]
        # results_t = []
        # for lst in results:
        #     results_t = list(chain(results_t, lst))
        # results = results_t

        x = list(map(lambda val: val * TIME_PER_READING, range(len(results))))
        chart_name = f'results_{processing}'

        path_to_save = CHARTS_PATH / Path(f'results')
        path_to_save.mkdir(parents=True, exist_ok=True)

        draw_plot(x1=x, y1=results, label1='Результат', chart_title='Результаты')


if __name__ == '__main__':
    pr = PredictMove()
    pr.process_different_moves_file(processing=ProcessingType.NETWORK)
