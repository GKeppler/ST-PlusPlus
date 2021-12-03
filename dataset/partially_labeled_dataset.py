from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class PartiallyLabeledDataset(Dataset, ABC):
    # @abstractmethod
    # def get_samples(self):
    #     raise NotImplementedError("This method needs to be implemented.")

    # @abstractmethod
    # def pop_sample(self, new_sample):
    #     raise NotImplementedError("This method needs to be implemented.")

    # @abstractmethod
    # def add_sample(self, index):
    #     raise NotImplementedError("This method needs to be implemented.")

    def set_raw_mode(self, raw_mode):
        self.raw_mode = raw_mode

    def resort_samples(self):
        self.indices = sorted(self.indices, key=lambda x: int(x))
