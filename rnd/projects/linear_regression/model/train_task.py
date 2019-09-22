from .linear import LinearModel
from common.job import JobConfig, TrainJob
from common.datasets.linear_data import DevDataset, TrainDataset


class Config(JobConfig):

    def add_arguments(self):
        self.parser.add_argument(
            '--epochs',
            help='GCS or local paths to training data',
            default=10000,
            required=False,
            type=int,
        )
        self.parser.add_argument(
            '--learning_rate', '--learning-rate',
            help='GCS or local paths to training data',
            default=0.001,
            required=False,
            type=float,
        )


def run_experiment(config):
    '''
    Google ML Engine entry point for training job.
    '''
    TrainJob(
        config=config,
        model_class=LinearModel,
        train_data_iterator=TrainDataset,
        dev_data_iterator=DevDataset,
    ).run()


if __name__ == '__main__':
    run_experiment(Config().parse())
