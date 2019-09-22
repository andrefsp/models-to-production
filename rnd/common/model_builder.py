""" Code for building models """

import tensorflow as tf


class Model(object):

    """
    Base model class

    ::
        All models on Neuron must subclass and implement this methods.
    """

    def __init__(self, config):
        self.config = config

    def get_callbacks(self, session):
        """ Returns a list of callbacks. Implementors can use this to send
        callbacks into Keras training methods, or for setting up callbacks for
        custom training methods (which will need to be manually handled).

        The session is provided for implementors to use if necessary.
        """
        return []

    def _prepare_export_path(self):
        for path, _, files in tf.gfile.Walk(self.config.export_path):
            for file in files:
                rm_path = (
                    '%s%s' % (path, file)
                    if path.endswith('/') else '%s/%s' % (path, file)
                )
                tf.gfile.Remove(rm_path)
        try:
            tf.gfile.DeleteRecursively(self.config.export_path)
        except tf.errors.NotFoundError:
            pass

    def build(self):
        """ Implementors should build the model, e.g. create the tensorflow
        graph or the compiled keras Model with this method. """

        raise NotImplementedError("Implement build() method")

    def train(self, session, *args, train_data_iterator=None,
              dev_data_iterator=None, **kwargs):
        """ Implementors should train the model with this method.

        Parameters
        ----------
        session : tf.Session
            The tensorflow session to use.

        train_data_iterator : Iterable
            Most implementors will provide the training data in the form of
            an iterable or iterator here.

        dev_data_iterator : Iterable
            Implementors should make this optional, and may allow a development
            set to be used during training via supplying the iterable here.
        """

        raise NotImplementedError("Implement train() method")

    def evaluate(self, session, *args, evaluate_data_iterator=None, **kwargs):
        """ Implementors should evaluate the model with this method.

        Parameters
        ----------
        session : tf.Session
            The tensorflow session to use.

        evaluate_data_iterator : Iterable
            Most implementors will provide the evaluation data in the form of
            an iterable or iterator here.
        """

        raise NotImplementedError("Implement evaluate() method")

    def predict(self, session, *args, predict_data_iterator=None, **kwargs):
        """ Implementors should evaluate the model with this method.

        Parameters
        ----------
        session : tf.Session
            The tensorflow session to use.

        predict_data_iterator : Iterable
            Most implementors will provide the prediction data in the form of
            an iterable or iterator here.
        """
        raise NotImplementedError("Implement predict() method")

    def save_tf_export(self, session):
        """ We use tensorflow serving for our production models, this method
        must save the model in the tensorflow serving format for the model to
        be usable with tensorflow serving."""
        raise NotImplementedError("Implement save_tf_export() method")

    def load_tf_export(self, session):
        """ This method must be implemented if we wish to restore a model
        saved in the tensorflow serving format for use in python. """
        raise NotImplementedError("Implement load_tf_export() method")

    def save_keras_model(self, session):
        """ This method is used to save in the keras .h5 format. If you
        train your model with Keras training, or evaluate using Keras methods,
        then you'll need to save it in this format to be able to restore the
        model and continue training or perform the evaluation. """

        raise NotImplementedError("Implement save_keras_model() method")

    def load_keras_model(self, session):
        """ This method is used to restore a model from the keras .h5 format.
        If you train your model with Keras training, or evaluate using Keras
        methods, then you'll need to save it in this format to be able to
        restore the model and continue training or perform the evaluation. """

        raise NotImplementedError("Implement save_keras_model() method")

    def save_checkpoint(self, session, checkpoint=None):
        saver = tf.train.Saver()
        saver.save(session, checkpoint or self.config.checkpoint)

    def load_checkpoint(self, session, checkpoint=None):
        saver = tf.train.Saver()
        saver.restore(session, checkpoint or self.config.checkpoint)
