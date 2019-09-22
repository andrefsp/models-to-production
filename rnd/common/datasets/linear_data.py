import os
import csv


SHARED_STORAGE = os.path.normpath(
    os.path.join(
        os.path.realpath(__file__),
        "..", "..", "..", "..", "shared-storage"
    )
)


class LinearDataIterator(object):
    def __iter__(self):
        return iter(csv.DictReader(open(self.filename)))


class EvaluationDataset(LinearDataIterator):
    filename = os.path.join(
        SHARED_STORAGE, "datasets/linear_data/linear.eval.csv"
    )


class DevDataset(LinearDataIterator):
    filename = os.path.join(
        SHARED_STORAGE, "datasets/linear_data/linear.dev.csv"
    )


class TrainDataset(LinearDataIterator):
    filename = os.path.join(
        SHARED_STORAGE, "datasets/linear_data/linear.train.csv"
    )
