"""RankNet on TensorFlow

Authors: NUKUI Shun<nukui.s@ai.cs.titech.ac.jp>
License: GNU ver.2.0

"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from six.moves import xrange
import numpy as np
import tensorflow as tf

ACTIVATE_FUNC = {"relu": tf.nn.relu,
                 "sigmoid": tf.nn.sigmoid}

class RankNet(object):
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
                 learning_rate=0.1, max_steps=1000, verbose=False,
                 logdir=None, q_capacity=1000000, min_after_dequeue=100,
                 threads=8):
        if not activate_func in ACTIVATE_FUNC:
            raise ValueError("'activate_func' must be in"
                            "['rele', 'sigmoid'")
        self.activate_func = activate_func
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.q_capacity = q_capacity
        self.max_steps = max_steps
        self.min_after_dequeue = min_after_dequeue
        self.threads = threads
        self.verbose = verbose
        self.batch_size = batch_size
        self.logdir = logdir

    def fit(self, data1, data2, pretraining=False, init=True):
        """learn the neural network
            The ranks of data1 are labeled higher than data2

            Arguments
            ---------------
            data1 : 2-D numpy array or Pandas DataFrame
                    which are higher rank than data2
            data2 : Must has the same shape as data1
        """
        shape1 = tuple(data1.shape)
        shape2 = tuple(data2.shape)
        if shape1 != shape2:
            raise ValueError("'data1' and 'data2' mush have the same shape")

        self.fdim = shape1[1]
        self.data_size = shape1[0]
        if init:
            self._setup_training()

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
        if self.logdir:
            writer = tf.train.SummaryWriter(self.logdir, sess.graph_def)
        #Run the training loop, controlling termination with the coord
        try:
            for step in xrange(self.max_steps):
                if coord.should_stop():
                    break
                cost, sm, _ = sess.run([self.cost, self.summary,
                                        self.optimize])
                if self.verbose and step%100:
                    print("The %dth cost:%f"%(step, cost))
                if self.logdir:
                    writer.add_summary(sm, step)
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(enqueue_threads)

    def _setup_training(self):
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            fdim = self.fdim
            batch_size = self.batch_size
            hidden_units = self.hidden_units
            layer_units = [fdim] + hidden_units + [1]
            layer_num = len(layer_units)
            act_func = ACTIVATE_FUNC[self.activate_func]
            lr = self.learning_rate

            #make Queue for getting batch
            self.queue = q = tf.RandomShuffleQueue(capacity=self.q_capacity,
                                        min_after_dequeue=self.min_after_dequeue,
                                        dtypes=["float", "float"],
                                        shapes=[[fdim], [fdim]])
            #input data
            data1, data2 = q.dequeue_many(batch_size, name="inputs")

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

            s1 = self._obtain_score(data1, weights, biases, act_func, 1)
            s2 = self._obtain_score(data2, weights, biases, act_func, 2)
            with tf.name_scope("cost"):
                self.cost = cost = tf.reduce_sum(
                                    tf.log(1 + tf.exp(-tf.nn.sigmoid(s1-s2))))
                optimizer = tf.train.AdamOptimizer(lr)
            self.optimize = optimizer.minimize(cost)

            tf.scalar_summary("cost", cost)
            self.summary = tf.merge_all_summaries()

            self.init_op = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(self.init_op)

    @classmethod
    def _obtain_score(cls, data, weights, biases, act_func, num):
        num_layer = len(weights) + 1
        outputs = [data]
        for n in xrange(num_layer-1):
            with tf.name_scope("layer"+str(n)+"_"+str(num)):
                input_n = outputs[n]
                w_n = weights[n]
                b_n = biases[n]
                output_n = act_func(tf.matmul(input_n, w_n) + b_n)
                outputs.append(output_n)
        score = outputs[-1]
        return score

    @classmethod
    def _get_weight_variable(cls, shape, layer, stddev=1.0):
        initializer = tf.random_normal_initializer(mean=0.0,
                                                   stddev=stddev)
        name = "weight" + str(layer)
        w = tf.get_variable(name=name, shape=shape,
                            dtype="float")
        return w

    @classmethod
    def _get_bias_variable(cls, shape, layer):
        init = tf.zeros_initializer(shape=shape)
        name = "bias" + str(layer)
        b = tf.Variable(init, name=name)
        return b

    def _set_weight(weight, layer):
        assign_op = self.weights[layer].assign(weight)
        self.sess.run(assign_op)

    def _set_bias(bias, layer):
        assign_op = self.biases[layer].assign(bias)
        self.sess.run(assign_op)
