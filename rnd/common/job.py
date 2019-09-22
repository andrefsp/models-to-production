import os
import logging
import sys
import json
import argparse
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

from .utils import normpath


def jobdir_path(job_dir):
    def _jobdir_path(path):
        return normpath('%s/%s' % (normpath(job_dir), path))
    return _jobdir_path


class ConfigException(Exception):
    pass


class JobConfig(object):

    """
    Default job configuration
    """

    _excluded = [
        '_model_structure', '_hparam_types', 'parser', '_args_dict', '_gfile',
        'load_from_config',
    ]

    _protected = [
        'export_path', 'log_path', 'checkpoint', 'keras_model',
        'config_file_path',
    ]

    _default_job_config = {
        'job_dir': './job/',
        'model_version': os.environ.get('VERSION', '-'),
        'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
        'slack_notify': os.environ.get(
            'SLACK_NOTIFY', 'false'
        ).lower() in ('true', '1'),
        'load_from': None,
    }
    export_path = None
    log_path = None
    keras_model = None
    load_from_config = None

    def __init__(self, gfile=tf.gfile):
        self.parser = self.get_argument_parser()
        self.add_arguments()
        self._gfile = gfile
        self._args_dict = {}

    @property
    def model_id(self):
        return "%s__%s" % (
            self.model_version,
            self.trial_id
        )

    @property
    def trial_id(self):
        # XXX(andrefsp):: Possible to get it from ML engine environment vars.
        # However that would make this dependent on Google. We dont want that!
        #
        # if os.environ.get('TF_CONFIG'):
        #     return json.loads(
        #         os.environ.get('TF_CONFIG', '{}')
        #     ).get('task', {}).get('trial', '0')

        # infer trial ID from job dir
        for part in reversed(self.job_dir.split('/')):
            if part and part.isdigit():
                return part
        return '0'

    def _load_from(self):
        """
            `_load_from` reloads the configuration from the previous model,
        copies the previous values onto the new configuration.
            Values passed on the current configuration will be kept and
        therefore overriding the previous values
        """
        self.load_from_config = type(self)().parse([
            '--job_dir', self.load_from,
        ]).from_json_file()

        for attr, val in self.load_from_config.__dict__.items():
            should_overwrite = all([
                attr not in self._excluded,
                attr not in self._args_dict,
                attr not in self._protected
            ])
            if should_overwrite:
                setattr(self, attr, val)

    def to_hparams(self, obj):

        _jobdir_path = jobdir_path(obj.job_dir)

        for key, val in obj.__dict__.items():
            if hasattr(self, key):
                raise ConfigException(
                    "Parser defines a argument already present on config."
                )
            setattr(self, key, val)

        self.export_path = _jobdir_path('export/')
        self.log_path = _jobdir_path('log/')
        self.checkpoint = _jobdir_path('checkpoint/model.ckpt')
        self.keras_model = _jobdir_path('keras_model.h5')
        self.config_file_path = _jobdir_path('export/config.json')

        for key, val in hparam.HParams(**self.__dict__).values().items():
            if key == "parser":
                continue
            setattr(self, key, val)

        if self.load_from is not None:
            self._load_from()

        return self

    def parse_known_args(self, args=None, **kwargs):
        if not args:
            args = sys.argv[1:]
        else:
            args = list(args)

        self._args_dict = self.parser._parse_known_args(
            args, argparse.Namespace()
        )[0].__dict__
        obj, _ = self.parser.parse_known_args(args, **kwargs)
        return self.to_hparams(obj)

    def parse(self, args=None, **kwargs):
        if not args:
            args = sys.argv[1:]
        else:
            args = list(args)

        self._args_dict = self.parser._parse_known_args(
            args, argparse.Namespace()
        )[0].__dict__
        return self.to_hparams(self.parser.parse_args(args, **kwargs))

    def add_arguments(self):
        pass

    def get_argument_parser(self):
        parser = argparse.ArgumentParser(prog='ml_job')

        # Arguments for both training, predicting and embeddings
        parser.add_argument(
            '--model-version', '--model_version',
            default=self._default_job_config['model_version'],
            help='path to model',
            required=False,
        )
        parser.add_argument(
            '--job-dir', '--job_dir',
            help='GCS or local paths to training data',
            default=self._default_job_config['job_dir'],
            required=True,
        )
        parser.add_argument(
            '--load-from', '--load_from',
            help='GCS or local paths to training data',
            default=self._default_job_config['load_from'],
            required=False,
        )
        parser.add_argument(
            '--log-level', '--log_level',
            help='Log level for the job',
            default=self._default_job_config['log_level'],
        )
        return parser

    def to_dict(self):
        dict = {}
        for a in dir(self):
            if a in self.__dict__ and a not in self._excluded:
                dict[a] = self.__getattribute__(a)
        return dict

    def from_dict(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)
        return self

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    def to_json_file(self):
        self._gfile.MakeDirs(self.export_path)
        with self._gfile.GFile(self.config_file_path, 'wb') as json_file:
            json_file.write(self.to_json())

    def from_json_file(self):
        with self._gfile.Open(self.config_file_path) as f:
            conf_str = f.read()
            if not conf_str:
                raise ConfigException("Could not read config json file")

        return self.from_dict(json.loads(conf_str))


class Job(object):

    """
    Main job
    """

    def __init__(self, config=None, model_class=None, report_class=None,
                 pre_job_hooks=None, post_job_hooks=None,
                 description='', *args, **kwargs):
        self.description = description
        self.config = config
        self.model_class = model_class
        self.report_class = report_class
        self.pre_job_hooks = pre_job_hooks or []
        self.post_job_hooks = post_job_hooks or []
        self.kwargs = kwargs
        self.args = args

    def get_summary_writer(self, graph):
        return tf.summary.FileWriter(self.config.log_path, graph)

    def get_job_kwargs(self, graph):
        kwargs = {}
        kwargs.update(self.kwargs)
        if 'summary_writer' not in kwargs:
            kwargs['summary_writer'] = self.get_summary_writer(graph)
        return kwargs

    @property
    def job_dir(self):
        return self.config.job_dir

    @property
    def version(self):
        return getattr(self.config, 'model_version', None)

    def get_job_info(self):
        return {
            'job_dir': self.job_dir,
            'version': self.version,
            'description': self.description
        }

    def _get_post_job_hooks(self):
        return self.post_job_hooks

    def _run_post_job_hooks(self, run_result):
        for trigger in self._get_post_job_hooks():
            trigger.run(self, run_result)

    def _run_pre_job_hooks(self):
        for trigger in self.pre_job_hooks:
            trigger.run(self)

    def _run(self):
        raise NotImplementedError("Error:: _run is not implemented")

    def create_report(self, model=None, *args, **kwargs):
        if not self.report_class:
            return

        self.report = self.report_class(
            model=model, *args, **kwargs
        )
        return self.report.create()

    def run(self):
        logging.basicConfig(
            stream=sys.stdout,
            level=getattr(logging, self.config.log_level)
        )
        self._run_pre_job_hooks()
        run_result = self._run()
        self._run_post_job_hooks(run_result)
        return run_result


class TrainJob(Job):

    """
    Train job abstraction
    """

    def get_model(self, session, config):
        model = self.model_class(config)
        model.build()

        session.run(tf.global_variables_initializer())

        if config.load_from_config:
            model.load_checkpoint(session, config.load_from_config.checkpoint)
        return model

    def _run(self):
        tf.reset_default_graph()
        graph = tf.get_default_graph()
        with graph.as_default() as graph:

            # export the job config in the beginning
            self.config.to_json_file()

            with tf.Session() as sess:
                model = self.get_model(sess, self.config)
                self.create_report(
                    model=model, job_kwargs=self.kwargs
                )
                result = model.train(
                    sess, *self.args, **self.get_job_kwargs(graph)
                )

            # export the job config at the end
            self.config.to_json_file()

        return result
