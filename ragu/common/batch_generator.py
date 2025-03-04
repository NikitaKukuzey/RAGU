import math
from typing import List, Generator


class BatchGenerator:
    """
    A utility class for generating batches of data.

    Attributes:
        data (List[str]): The dataset to be batched.
        batch_size (int): The size of each batch.
    """

    def __init__(self, data: List[str], batch_size: int):
        """
        Initializes the BatchGenerator.

        :param data: A list of strings representing the dataset.
        :param batch_size: The number of elements in each batch.
        """
        self.data = data
        self.batch_size = batch_size

    def get_batches(self) -> Generator:
        """
        Generates batches from the dataset.

        :return: A generator that yields batches of data.
        """
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i : i + self.batch_size]

    def __len__(self) -> int:
        """
        Returns the number of batches.

        :return: The total number of batches.
        """
        return math.ceil(len(self.data) / self.batch_size)
