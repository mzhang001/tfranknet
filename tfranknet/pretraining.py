"""Autoencoder on Tensorflow"""

#Authors: Nukui Shun <nukui.s@ai.cs.titech.ac.jp>
#License : GNU General Public License v2.0

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from six.moves import xrange

import numpy as np
import tensorflow as tf

ACTIVATE_FUNC = {"relu": tf.nn.relu,
                "sigmoid": tf.nn.sigmoid,
                "softplus": tf.nn.softplus}

OPTIMIZER = {"sgd": tf.train.GradientDescentOptimizer,
            "adagrad": tf.train.AdagradOptimizer,
            "adam": tf.train.AdamOptimizer}

class TFAutoEncoder(object):
    """class for Auto Encoder on Tensorflow

        Attributes (hidden_dim)
        --------------------
        hidden_dim : The number of units in hidden layer

        learning_rate : Learning rate used in optimization

        noising : If True fitting as denoising autoencoder(defalut is False)

        noise_stddev : Only used when noising is True

        activate_func : Selected in 'relu'(default) 'softplus' 'sigmoid'

        optimizer : Selected in 'sgd' 'adagrad' 'adam'(default)

        num_epoch : The number of epochs in optimization

        w_stddev : Used in initializing weight variables

        lambda_w : Penalty coefficient of regularization term

        num_cores : The number of cores used in Tensorflow computation

        logdir : Directory to export log

        continue_training : If True keep the learned state at each time
                                        performing fit()
    """

    def __init__(self, hidden_dim, learning_rate=0.01, noising=False,
                noise_stddev=10e-2, activate_func="relu", optimizer="adam",
                num_epoch=100, w_stddev=0.1, lambda_w=0, num_cores=4,
                logdir=None, continue_training=False):
        if not activate_func in ACTIVATE_FUNC:
            raise ValueError("activate_func must be chosen of the following:"
                            "'relu','sigmoid','softplus'")
        if not optimizer in OPTIMIZER:
            raise ValueError("optimizer must be chosen of the following:"
                            "'sgd','adagrad','adam'")
        self.activate_func = activate_func
        self.optimizer = optimizer
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.noising = noising
        self.num_epoch = num_epoch
        self.num_cores = num_cores
        self.w_stddev = w_stddev
        self.noise_stddev = noise_stddev
        self.lambda_w = lambda_w
        self.continue_training = continue_training
        self.logdir = logdir
        self._initialized = False

    def fit(self, data):
        """Optimize autoencoder"""
        data = np.array(data)
        shape = data.shape
        if len(shape) != 2:
            raise TypeError("The shape of data is invalid")
        self.input_dim = shape[1]

        #setup computational graph if not initialized
        if not (self.continue_training and self._initialized):
            self._setup_graph()
        sess = self.session

        #setup summary writer for TensorBoard
        if self.logdir:
            writer = tf.train.SummaryWriter(self.logdir, sess.graph_def)

        for step in xrange(self.num_epoch):
            feed_dict = self._get_feed_dict(data, self.noising)
            l2_loss, summ, _ = sess.run([self._l2_loss, self._summ, self._optimize],
                                                    feed_dict=feed_dict)
            if self.logdir:
                writer.add_summary(summ, step)
        self.fit_loss = l2_loss


    def encode(self, data):
        """Encode data by learned autoencoder """
        sess = self.session
        feed_dict = self._get_feed_dict(data, noising=False)
        encoded = sess.run(self._encoded,
                          feed_dict=feed_dict)
        return encoded

    def reconstruct(self, data):
        """Encode and decode input data"""
        sess = self.session
        feed_dict = self._get_feed_dict(data, noising=False)
        reconstructed = sess.run(self._reconstructed,
                                feed_dict=feed_dict)
        return reconstructed

    def _get_feed_dict(self, data, noising):
        shape = data.shape
        feed_dict = {self._input: data,
                    self._batch_size: float(shape[0])}
        if noising:
            noise = self._generate_noise(shape, self.noise_stddev)
            feed_dict[self._noise] = noise
        else:
            zeros = np.zeros(shape=shape)
            feed_dict[self._noise] = zeros
        return feed_dict

    def _setup_graph(self):
        """Setup computation graph for training"""
        self._graph = tf.Graph()
        with self._graph.as_default():
            input_dim, hidden_dim = self.input_dim, self.hidden_dim
            lr = self.learning_rate
            activate_func = ACTIVATE_FUNC[self.activate_func]
            optimizer = OPTIMIZER[self.optimizer]

            self._input = X = tf.placeholder(name="X", dtype="float",
                                            shape=[None, input_dim])
            self._batch_size = batch_size = tf.placeholder(name="batchsize",
                                                            dtype="float")
            self._noise = noise = tf.placeholder(name="noise", dtype="float",
                                                shape=[None, input_dim])
            clean_X = X
            X = X + noise

            self._W = W = self._weight_variable(shape=[input_dim, hidden_dim],
                                                stddev=self.w_stddev)
            #bias in bottom layer
            self._b = b = self._bias_variable([hidden_dim])
            #bias in upper layer
            self._c = c = self._bias_variable([input_dim])

            encoded = activate_func(tf.matmul(X, W) + b)
            self._encoded = encoded

            Wt = tf.transpose(W)
            reconstructed = activate_func(tf.matmul(encoded, Wt) + c)
            self._reconstructed = reconstructed

            regularizer = self.lambda_w * tf.nn.l2_loss(W)
            l2_loss = tf.nn.l2_loss(clean_X - reconstructed) / batch_size
            self._l2_loss = l2_loss
            self._loss = loss = l2_loss + regularizer
            self._optimize = optimizer(lr).minimize(loss)

            #variables summary
            tf.scalar_summary("l2_loss", l2_loss)
            tf.scalar_summary("loss", loss)
            self._summ = tf.merge_all_summaries()

            #create session
            self.session = tf.Session(config=tf.ConfigProto(
                                    inter_op_parallelism_threads=self.num_cores,
                                    intra_op_parallelism_threads=self.num_cores))
            #create initializer
            self._initializer = tf.initialize_all_variables()
            self.session.run(self._initializer)
            self._initialized = True

    @classmethod
    def _weight_variable(cls, shape, stddev):
        """Generate weight matrix variable by normal distribution"""
        init = tf.truncated_normal(shape, stddev)
        w = tf.Variable(init)
        return w

    @classmethod
    def _bias_variable(cls, shape):
        """Generate bias vector variable by ones"""
        #init = tf.truncated_normal(shape, stddev)
        init = tf.ones(shape)
        b = tf.Variable(init)
        return b

    @classmethod
    def _generate_noise(cls, shape, stddev):
        """Generate noise for denoising autoencoder"""
        noise = np.random.normal(size=shape, scale=stddev)
        return noise

    @property
    def weight(self):
        """Return weight as numpy array"""
        sess = self.session
        W = sess.run(self._W)
        return W

    @property
    def bias(self):
        """Return bias as numpy array"""
        sess = self.session
        b = sess.run(self._b)
        return b

    @property
    def bias_upper(self):
        """Return bias as numpy array"""
        sess = self.session
        c = sess.run(self._c)
        return c

class TFStackedAutoEncoder(object):
    """Class for Stacked Auto Encoder on Tensorflow

        Attributes (layer_units)
        --------------------
        layer_units : The number of units in each layer.

        learning_rate : Learning rate used in optimization

        noising : If True fitting as denoising autoencoder(defalut is False)

        noise_stddev : Only used when noising is True

        activate_func : Selected in 'relu'(default) 'softplus' 'sigmoid'

        optimizer : Selected in 'sgd' 'adagrad' 'adam'(default)

        num_epoch : The number of epochs in optimization

        w_stddev : Used in initializing weight variables

        lambda_w : Penalty coefficient of regularization term

        num_cores : The number of cores used in Tensorflow computation

        logdir : Directory to export log
    """

    def __init__(self, layer_units, learning_rate=0.01, noising=False,
                noise_stddev=10e-2, activate_func="relu", optimizer="adam",
                num_epoch=100, w_stddev=0.1, lambda_w=0, num_cores=4,
                logdir=None):
        self.layer_units = layer_units
        self.layer_num = len(layer_units)
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.num_cores = num_cores
        self.w_stddev = w_stddev
        self.lambda_w = lambda_w
        self.noising = noising
        self.noise_stddev = noise_stddev
        self.activate_func = activate_func
        self.optimizer = optimizer
        self.logdir = logdir

    def fit(self, data):
        """Optimize stacked autoencoder from the bottom layer"""
        data = np.array(data)
        shape = data.shape
        if len(shape) != 2:
            raise TypeError("The shape of data is invalid")
        if shape[1] != self.layer_units[0]:
            raise ValueError("Input dimension must match to 1st layer units")
        outputs = [data]
        weight = []
        bias = []
        for n in xrange(1, self.layer_num):
            input_n = outputs[n-1]
            hidden_dim = self.layer_units[n]
            output_n, W_n, b_n = self._partial_fit(input_n, hidden_dim, n)
            outputs.append(output_n)
            weight.append(W_n)
            bias.append(b_n)
        self._setup_encode(weight, bias)

    def encode(self, data, layer=-1):
        """Encode data by learned stacked autoencoder """
        sess = self.session
        encoded = sess.run(self._outputs[layer],
                        feed_dict={self._input: data})
        return encoded

    def _setup_encode(self, weight_, bias_):
        """Setup computation graph for encode
           Initialize weights and biases to given values
        """
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._input = X = tf.placeholder(dtype="float",
                                            shape=[None, self.layer_units[0]])

            #setup data nodes for weights and biases
            weight = []
            bias = []
            for n, w in enumerate(weight_):
                with tf.variable_scope("layer"+str(n)):
                    weight.append(tf.Variable(w, name="weight"))
            for n, b in enumerate(bias_):
                with tf.variable_scope("layer"+str(n)):
                    bias.append(tf.Variable(b, name="bias"))
            self._weight = weight
            self._bias = bias

            #define encoder: outputs[-1] is final result of encoding
            self._outputs = outputs = [X]
            activate_func = ACTIVATE_FUNC[self.activate_func]
            for n in xrange(self.layer_num-1):
                x = outputs[n]
                w = weight[n]
                b = bias[n]
                output_n = activate_func(tf.matmul(x, w) + b)
                outputs.append(output_n)

            #create session
            self.session = tf.Session(config=tf.ConfigProto(
                                    inter_op_parallelism_threads=self.num_cores,
                                    intra_op_parallelism_threads=self.num_cores))
            init_op = tf.initialize_all_variables()
            self.session.run(init_op)

    def _partial_fit(self, data, hidden_dim, layer_no):
        """Optimize single autoencoder"""
        #make logdir for each layer
        if self.logdir:
            logdir = self.logdir + "/layer" + str(layer_no)
            if not os.path.exists(logdir):
                os.makedirs(logdir)
        else:
            logdir = self.logdir
        ae = TFAutoEncoder(hidden_dim=hidden_dim,
                        learning_rate=self.learning_rate,
                        noising=self.noising,
                        noise_stddev=self.noise_stddev,
                        w_stddev=self.w_stddev,
                        lambda_w=self.lambda_w,
                        num_epoch=self.num_epoch,
                        activate_func=self.activate_func,
                        optimizer=self.optimizer,
                        continue_training=False,
                        logdir=logdir,
                        num_cores=self.num_cores)
        ae.fit(data)
        output = ae.encode(data)
        weight = ae.weight
        bias = ae.bias
        return output, weight, bias

    @property
    def weight(self):
        sess = self.session
        wlist = []
        for w in self._weight:
            wlist.append(sess.run(w))
        return np.array(wlist)

    @property
    def bias(self):
        sess = self.session
        blist = []
        for b in self._bias:
            blist.append(sess.run(b))
        return np.array(blist)
