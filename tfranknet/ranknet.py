"""RankNet on TensorFlow"""

#Authors: NUKUI Shun<nukui.s@ai.cs.titech.ac.jp>
#License : GNU General Public License v2.0

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import pickle
from six.moves import xrange

import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf

ACTIVATE_FUNC = {"relu": tf.nn.relu,
                 "sigmoid": tf.nn.sigmoid}

OPTIMIZER = {"sgd": tf.train.GradientDescentOptimizer,
             "adam": tf.train.AdamOptimizer}

class RankNet(BaseEstimator):
    """Class for RankNet

        References
        ------------
        Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds,
        Nicole Hamilton, and Greg Hullender. 2005.
        Learning to rank using gradient descent.
        In Proceedings of the 22nd international conference on Machine learning
        (ICML '05).ACM, New York, NY, USA, 89-96.
    """

    def __init__(self, hidden_units, batch_size=32, activate_func="relu",
                 learning_rate=0.01, max_steps=1000, sigma=1.0,
                 q_capacity=1000000, min_after_dequeue=100, optimizer="sgd",
                 threads=8, verbose=False, initialize=True):
        if not activate_func in ACTIVATE_FUNC:
            raise ValueError("'activate_func' must be in"
                            "['rele', 'sigmoid'")
        self.activate_func = activate_func
        self.hidden_units = hidden_units
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.q_capacity = q_capacity
        self.max_steps = max_steps
        self.min_after_dequeue = min_after_dequeue
        self.threads = threads
        self.verbose = verbose
        self.batch_size = batch_size
        self.layer_num = len(hidden_units) + 2
        self.optimizer = optimizer
        self.initialize = initialize
        self.initialized = False


    def fit(self, data, logdir=None, label=None):
        """Learn the ranking neural network
            The ranks of data1[i] are labeled higher than data2[i]

            The input data must be packed by pack_data(data1, data2)
                before calling this function

            Arguments
            ---------------
            datas : A list of 2-D numpy array or Pandas DataFrame
                    such that datas[0][i] are higher rank than data[1][i]
            pretraining : If True pretrain by AutoEncoder(Not implemented)
            init : If True initialize the graph and variables
            label : A dummy argument for cross validation in skleran
        """
        data1, data2 = self.unpack_data(data)
        shape1 = tuple(data1.shape)
        shape2 = tuple(data2.shape)

        self.input_dim = shape1[1]
        self.data_size = shape1[0]
        if self.initialize or (not self.initialized):
            #prepare a training graph and initialize
            self.initialize_graph(self.input_dim)
        #fine tuning
        self._fine_tuning(data1, data2, logdir)

    def initialize_graph(self, input_dim):
        self.input_dim = input_dim
        self._setup_base_graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.init_op = tf.initialize_all_variables()
            self.summary = tf.merge_all_summaries()
            self.sess.run(self.init_op)
        self.initialized = True

    def pretrain(self, data):
        input_dim = data.shape[1]
        data_size = data.shape[0]
        self.initialize_graph(input_dim)
        layer_num = self.layer_num
        max_steps = self.max_steps
        batch_size = self.batch_size
        sess = self.sess
        pretrain_layer = self.pretrain_layer
        for n in xrange(layer_num-2):
            np.random.shuffle(data)
            #train loop
            for step in xrange(max_steps):
                start = step * batch_size
                stop = (step+1) * batch_size
                if start >= data_size:
                    break
                batch = data[start: stop]
                sess.run([pretrain_layer[n]],
                        feed_dict={self.pt_input: batch})
            data = sess.run(self.encode[n], feed_dict={self.pt_input: data})

    def _fine_tuning(self, data1, data2, logdir):
        #set data to enqueue op(not executed yet)
        data1 = np.array(data1)
        data2 = np.array(data2)
        with self.graph.as_default():
            enq = self.queue.enqueue_many((data1, data2))
        #Not used after defining enqueue op
        del data1
        del data2

        qr = tf.train.QueueRunner(self.queue, [enq]*self.threads)
        sess = self.sess
        coord = tf.train.Coordinator()
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
        if logdir:
            writer = tf.train.SummaryWriter(logdir, sess.graph_def)
        #Run the training loop, controlling termination with the coord
        try:
            for step in xrange(self.max_steps):
                if coord.should_stop():
                    break
                cost, sm, _ = sess.run([self.cost, self.summary,
                                        self.optimize])
                if self.verbose and step%100:
                    print("The %dth cost:%f"%(step, cost))
                if logdir:
                    writer.add_summary(sm, step)
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(enqueue_threads)

    def predict(self, data):
        """Predict whether data1[i] is higher rank than data2[i]
        """
        prob = self.predict_prob(data)
        pred = (prob > 0.5)
        return pred

    def predict_prob(self, data):
        """
        Predict probabilities that data1[i] is higher rank than data2[i]
        """
        data1, data2 = self.unpack_data(data)
        sess = self.sess
        feed_dict = {self.input1: data1, self.input2: data2}
        prob = sess.run(self.prob, feed_dict=feed_dict)
        return prob

    def get_scores(self, data):
        """
        Compute scores for a unpacked data
        """
        feed_dict = {self.input1: data}
        scores = self.sess.run(self.score, feed_dict=feed_dict)
        return scores

    def _setup_base_graph(self):
        """
        Set up queue, variables and session
        """
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            input_dim = self.input_dim
            batch_size = self.batch_size
            hidden_units = self.hidden_units
            layer_units = [input_dim] + hidden_units + [1]
            layer_num = len(layer_units)

            #make Queue for getting batch
            self.queue = q = tf.RandomShuffleQueue(capacity=self.q_capacity,
                                        min_after_dequeue=self.min_after_dequeue,
                                        dtypes=["float", "float"],
                                        shapes=[[input_dim], [input_dim]])
            #input data
            self.data1, self.data2 = q.dequeue_many(batch_size, name="inputs")

            self._setup_variables()
            self._setup_training()
            self._setup_prediction()
            self._setup_pretraining()


    def _setup_variables(self):
        hidden_units = self.hidden_units
        layer_units = [self.input_dim] + hidden_units + [1]
        layer_num = len(layer_units)
        #setting weights and biases
        self.weights = weights = []
        self.biases = biases = []
        for n in xrange(layer_num-1):
            w_shape = [layer_units[n], layer_units[n+1]]
            b_shape = [layer_units[n+1]]
            w = self._get_weight_variable(w_shape, n)
            b = self._get_bias_variable(b_shape, n)
            weights.append(w)
            biases.append(b)

    def _setup_training(self):
        """
        Set up a data flow graph for fine tuning
        """
        layer_num = self.layer_num
        act_func = ACTIVATE_FUNC[self.activate_func]
        sigma = self.sigma
        lr = self.learning_rate
        weights = self.weights
        biases = self.biases
        data1, data2 = self.data1, self.data2
        batch_size = self.batch_size
        optimizer = OPTIMIZER[self.optimizer]
        with tf.name_scope("training"):
            s1 = self._obtain_score(data1, weights, biases, act_func, "1")
            s2 = self._obtain_score(data2, weights, biases, act_func, "2")
            with tf.name_scope("cost"):
                sum_cost = tf.reduce_sum(tf.log(1 + tf.exp(-sigma*(s1-s2))))
                self.cost = cost = sum_cost / batch_size
        self.optimize = optimizer(lr).minimize(cost)

        for n in range(layer_num-1):
            tf.histogram_summary("weight"+str(n), weights[n])
            tf.histogram_summary("bias"+str(n), biases[n])
        tf.scalar_summary("cost", cost)

    def _setup_pretraining(self):
        """
        Set up a data flow graph for pretraining by sAE
        """
        input_dim = self.input_dim
        weights = self.weights
        biases = self.biases
        layer_num = self.layer_num
        lr = self.learning_rate
        act_func = ACTIVATE_FUNC[self.activate_func]
        optimizer = OPTIMIZER[self.optimizer]

        self.pt_input = input_ = tf.placeholder("float", shape=[None, None],
                                                name="input_to_ae")
        self.pretrain_layer = []
        self.encode = []
        with tf.name_scope("pretraing"):
            for n in xrange(layer_num-2):
                w = weights[n]
                b = biases[n]
                label = str(n)
                encoded, recon_err = self._get_reconstruction_error(input_,
                                                                    w, b,
                                                                    act_func,
                                                                    label)
                opt_op = optimizer(lr).minimize(recon_err)
                self.pretrain_layer.append(opt_op)
                self.encode.append(encoded)

    def _setup_prediction(self):
        input_dim = self.input_dim
        self.input1 = inp1 = tf.placeholder("float", shape=[None,input_dim],
                                            name="input1")
        self.input2 = inp2 = tf.placeholder("float", shape=[None,input_dim],
                                            name="input2")
        weights = self.weights
        biases = self.biases
        act_func = ACTIVATE_FUNC[self.activate_func]
        sigma = self.sigma
        with tf.name_scope("prediction"):
            s1 = self._obtain_score(inp1, weights, biases, act_func, "1")
            s2 = self._obtain_score(inp2, weights, biases, act_func, "2")
            self.score = s1
            with tf.name_scope("probability"):
                self.prob = 1 / (1 + tf.exp(-sigma*(s1-s2)))

    @classmethod
    def _get_reconstruction_error(cls, input_, weight, bias, act_func, label):
        """
        Get encoded data and reconstruction error
        """
        with tf.variable_scope(label):
            b_shape = weight.get_shape()[0:1]
            #Only used for decoding
            bias_aux = cls._get_bias_variable(b_shape, "_aux_"+label)
        with tf.name_scope("AE"+label):
            encoded = act_func(tf.matmul(input_, weight) + bias)
            w_T = tf.transpose(weight)
            decoded = act_func(tf.matmul(encoded, w_T) + bias_aux)
        loss = tf.nn.l2_loss(input_ - decoded) + tf.nn.l2_loss(weight)
        return encoded, loss




    @classmethod
    def _obtain_score(cls, data, weights, biases, act_func, label):
        num_layer = len(weights) + 1
        outputs = [data]
        for n in xrange(num_layer-1):
            with tf.name_scope("layer"+str(n)+"_"+label):
                input_n = outputs[n]
                w_n = weights[n]
                b_n = biases[n]
                output_n = act_func(tf.matmul(input_n, w_n) + b_n)
                outputs.append(output_n)
        score = outputs[-1]
        return score

    @classmethod
    def _get_weight_variable(cls, shape, label, stddev=1.0):
        initializer = tf.random_normal_initializer(mean=0.0,
                                                   stddev=stddev)
        name = "weight" + str(label)
        w = tf.get_variable(name=name, shape=shape,
                            dtype="float")
        return w

    @classmethod
    def _get_bias_variable(cls, shape, label):
        init = tf.zeros_initializer(shape=shape)
        name = "bias" + str(label)
        b = tf.Variable(init, name=name)
        return b


    def set_weight(self, weight, layer=None):
        if layer is None:
            assign_op = [self.weights[n].assign(weight[n])
                              for n in xrange(self.layer_num-1)]
        else:
            assign_op = self.weights[layer].assign(weight)
        self.sess.run(assign_op)

    def set_bias(self, bias, layer=None):
        if layer is None:
            assign_op = [self.biases[n].assign(bias[n])
                              for n in xrange(self.layer_num-1)]
        else:
            assign_op = self.biases[layer].assign(bias)
        self.sess.run(assign_op)

    def get_weight(self, layer=None):
        if layer is None:
            weights = [self.sess.run(self.weights[n])
                        for n in xrange(self.layer_num-1)]
            return weights
        else:
            return self.sess.run(self.weights[layer])

    def get_bias(self, layer=None):
        if layer is None:
            biases = [self.sess.run(self.biases[n])
                       for n in xrange(self.layer_num-1)]
            return biases
        else:
            return self.sess.run(self.biases[layer])

    @staticmethod
    def unpack_data(data):
        """
        Unpack a packed data
        """
        e = TypeError("'data' must be packed by pack_data() at the beginning")
        try:
            shape = data.shape
        except Exception:
            raise e
        if (len(shape) != 3) or (shape[1] != 2):
            raise e
        data1 = data[:,0,:]
        data2 = data[:,1,:]
        return data1, data2

    @staticmethod
    def pack_data(data1, data2):
        """Pack two data for preprocessing
            By this function cross_validation() in sklearn can be applied
            for this class
        """
        e = TypeError("'data1' and 'data2' must be 2-D numpy array or"
                    "Pandas DataFrame such that data1.shape == data2.shape")
        try:
            shape1 = tuple(data1.shape)
            shape2 = tuple(data2.shape)
        except Exception:
            raise e
        if (shape1 != shape2):
            raise e
        data1 = np.array(data1)
        data2 = np.array(data2)
        packed = np.array([data1, data2]).swapaxes(0,1)
        return packed
