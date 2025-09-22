from abc import ABC, abstractmethod
from pathlib import Path
from rmgen_ds.parameters_scene import ParametersScene


class DSGenerator(ABC):
    def __init__(
        self,
    ):
        self.parameters: ParametersScene = []

    @abstractmethod
    def convert(self, entry):
        pass