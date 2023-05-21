from typing import Iterable

from constants import MoveTypes
from settings import THRESHOLD_FUNCTION_VALUES


class PredictMove:

    @staticmethod
    def _is_overcome_thresholds(*, vector: list[float], thresholds: dict[int, float]) -> bool:
        for index in range(len(vector)):
            if vector[index] < thresholds[index]:
                return False

        return True

    def predict_by_threshold(self, *, vector: list[float]) -> int:
        for move, thresholds in THRESHOLD_FUNCTION_VALUES.items():
            if self._is_overcome_thresholds(vector=vector, thresholds=thresholds):
                return MoveTypes.NUMS[move]

        return MoveTypes.NUMS[MoveTypes.REST]
