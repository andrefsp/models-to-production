import tensorflow as tf
import numpy as np
from common.model_builder import Model


class LinearModel(Model):

    def build(self):
        # Inputs
        self.X = tf.placeholder(tf.float32, name='X')
        self.Y = tf.placeholder(tf.float32, name='Y')

        # Variables to optimize
        self.M = tf.Variable(np.random.rand(), name='M')
        self.B = tf.Variable(np.random.rand(), name='B')

        # model:  Y = M*X + B
        self.model = tf.add(tf.multiply(self.M, self.X), self.B, name='model')

        # Model save and serving signature map
        self.tensor_info_X = tf.saved_model.utils.build_tensor_info(self.X)
        self.tensor_info_model = tf.saved_model.utils.build_tensor_info(
            self.model
        )
        self.signature_def_map = {
            'serving_default': (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'X': self.tensor_info_X},
                    outputs={'model': self.tensor_info_model},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            ),
        }

        # Mean squared error
        # Compare the output of the model with training 'Y'
        self.ms_error = tf.reduce_sum(
            tf.pow(self.model - self.Y, 2), name='ms_error'
        )

        # Minimize the Mean Squared Error with an Adam Optimizer
        self.training = tf.train.AdamOptimizer(
            self.config.learning_rate
        ).minimize(self.ms_error)

    def save_tf_export(self, session):
        self._prepare_export_path()

        # save builder
        builder = tf.saved_model.builder.SavedModelBuilder(
            self.config.export_path
        )
        # Save the model
        builder.add_meta_graph_and_variables(
            session,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map=self.signature_def_map,
        )
        builder.save()

    def evaluate(self, session, eval_data_iterator, summary_writer=None):
        x_train = []
        y_train = []
        for row in eval_data_iterator():
            x_train.append(row['X'])
            y_train.append(row['Y'])
        return {
            'error': session.run(
                self.ms_error,
                feed_dict={self.X: x_train, self.Y: y_train},
            )
        }

    def train(self, session, train_data_iterator, dev_data_iterator,
              summary_writer=None):
        session.run(tf.global_variables_initializer())

        print("start:\t Y = %s*X + %s" % (
            session.run(self.M), session.run(self.B))
        )
        for epoch in range(self.config.epochs):
            if epoch % 100 == 0:
                print("\t Epoch %s ::: \tY = %s*X + %s" % (
                    epoch, session.run(self.M), session.run(self.B))
                )

                # run evaluation at the each 100 epochs...
                print("\t\t %s" % self.evaluate(session, dev_data_iterator))

            x_train = []
            y_train = []
            for row in train_data_iterator():
                x_train.append(row['X'])
                y_train.append(row['Y'])

            session.run(
                self.training,
                feed_dict={self.X: x_train, self.Y: y_train}
            )

        self.save_tf_export(session)

    def load_tf_export(self, session):
        # Load saved model
        tf.saved_model.loader.load(
            session, [
                tf.saved_model.tag_constants.SERVING
            ],
            self.config.export_path
        )

        graph = tf.get_default_graph()

        # Inputs
        self.Y = graph.get_tensor_by_name("Y:0")
        self.X = graph.get_tensor_by_name("X:0")

        # variables to optimise
        self.M = graph.get_tensor_by_name("M:0")
        self.B = graph.get_tensor_by_name("B:0")

        # model
        self.model = graph.get_tensor_by_name("model:0")

        # ms_error
        self.ms_error = graph.get_tensor_by_name("ms_error:0")
