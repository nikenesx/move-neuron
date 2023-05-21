from typing import Iterable

import numpy as np

from constants import MoveTypes
from data_process import DataProcess
from neural_network import Network
from settings import THRESHOLD_FUNCTION_VALUES


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
        """Определить тип движения через пороговую функцию"""
        for move, thresholds in THRESHOLD_FUNCTION_VALUES.items():
            if self._is_overcome_thresholds(vector=vector, thresholds=thresholds):
                return MoveTypes.NUMS[move]

        return MoveTypes.NUMS[MoveTypes.REST]

    def predict_by_network(self, *, vector: list[float]) -> int:
        """Определить тип движения через нейронную сеть"""
        vector = np.expand_dims(vector, axis=0)
        return self.network.TRAINED_MODEL.predict(vector, verbose=0)
