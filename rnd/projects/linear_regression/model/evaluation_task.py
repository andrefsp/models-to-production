from .linear import LinearModel
from common.jobs.job import JobConfig
from common.jobs.evaluate import EvaluateJob


class Config(JobConfig):

    def add_arguments(self):
        self.parser.add_argument(
            '--eval-file', '--eval_file',
            help='GCS or local paths to training data',
            default="model/data/linear.train.csv",
            required=False
        )


def run_experiment(config):
    '''
    Google ML Engine entry point for training job.
    '''
    print(EvaluateJob(
        config=config,
        model_class=LinearModel,
        eval_data_iterator=config.eval_file,
    ).run())


if __name__ == '__main__':
    run_experiment(Config().parse())
