# coding=utf-8
from sklearn.metrics import confusion_matrix
import sklearn as sk
import tensorflow as tf
import numpy as np
import os


class CNNConfig(object):
    """
    # TODO: 在此修改TextCNN以及训练的参数
    """
    def __init__(self, train_mode='CHAR-RANDOM'):
        self.train_mode = train_mode  # 训练模式，'CHAR-RANDOM'为字符级，随机初始化词向量并训练优化
        # 'WORD-NON-STATIC'为词级, 使用word2vec预训练词向量并能够继续在训练中优化
        # 'MULTI'
        self.class_num = 1258  # 输出类别的数目
        self.embedding_dim = 128  # 词向量维度，仅'CHAR-RANDOM'模式适用，
        # 'WORD-NON-STATIC'模式默认为preprocess.py中定义的vec_dim

        self.filter_num = 200  # 卷积核数目
        self.filter_sizes = [2, 3, 4, 5]  # 卷积核尺寸

        self.dense_unit_num = 512  # 全连接层神经元

        self.dropout_keep_prob = 0.3  # dropout保留比例
        self.learning_rate = 1e-3 # 学习率

        self.train_batch_size = 128  # 每批训练大小
        self.valid_batch_size = 3000  # 每批验证大小
        self.test_batch_size = 5000  # 每批测试大小
        self.valid_per_batch = 1000  # 每多少批进行一次验证
        self.epoch_num = 26  # 总迭代轮次


class TextCNN(object):
    def __init__(self, config):
        self.class_num = config.class_num
        self.filter_sizes = config.filter_sizes
        self.filter_num = config.filter_num

        self.dense_unit_num = config.dense_unit_num
        self.train_batch_size = config.train_batch_size
        self.valid_batch_size = config.valid_batch_size
        self.test_batch_size = config.test_batch_size


        self.text_length = 12
        self.embedding_dim = 100

        self.train_mode = config.train_mode

        self.input_x = None
        self.input_y = None
        self.labels = None
        self.dropout_keep_prob = None
        self.training = None
        self.embedding_inputs_expanded = None
        self.loss = None
        self.accuracy = None
        self.prediction = None
        self.vocab = None
        self.vecs_dict = {}
        self.embedding_W = None

    def setCNN(self):
        # 输入层
        self.input_x = tf.placeholder(tf.int32, [None, self.text_length], name="input_x")
        self.labels = tf.placeholder(tf.int32, [None], name="input_y")
        # 把数字标签转为one hot形式
        self.input_y = tf.one_hot(self.labels, self.class_num)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 训练时batch_normalization的Training参数应为True,
        # 验证或测试时应为False
        self.training = tf.placeholder(tf.bool, name='training')

        # 词嵌入层
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # 用之前读入的预训练词向量
            W = tf.Variable(self.embedding_W)
            embedding_inputs = tf.nn.embedding_lookup(W, self.input_x)
            print(embedding_inputs)
            self.embedding_inputs_expanded = tf.expand_dims(embedding_inputs, -1)

        # The final pooling output, containing outputs from each filter
        pool_outputs = []
        # Iterate to create convolution layer for each filter
        for filter_size in self.filter_sizes:
            with tf.name_scope("conv-maxpool-%d" % filter_size):
                # Convolution layer 1
                # ==================================================================
                # To perform conv2d, filter param should be [height, width, in_channel, out_channel]
                filter_shape = [filter_size, self.embedding_dim]
                print(self.embedding_inputs_expanded)
                conv_1 = tf.layers.conv2d(
                    inputs=self.embedding_inputs_expanded,
                    filters=self.filter_num,
                    kernel_size=filter_shape,
                    strides=[1, 1],
                    padding='VALID',
                    use_bias=True,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    bias_initializer=tf.constant_initializer(0.1)
                )
                # ===================================================================
                # Do batch normalization
                # =================================================================
                conv_1_output = tf.layers.batch_normalization(conv_1, training=self.training)
                conv_1_output = tf.nn.relu(conv_1_output)
                # ======================================================================
                # Pooling layer 1
                # ====================================================================
                conv_1_output_shape = conv_1_output.shape.as_list()
                pool_1 = tf.layers.max_pooling2d(
                    inputs=conv_1_output,
                    pool_size=[conv_1_output_shape[1] - 1 + 1, 1],
                    strides=[1, 1],
                    padding='VALID'
                )
                # =====================================================================

            pool_outputs.append(pool_1)

        # Combine all the pooling output
        # The total number of filters.
        total_filter_num = self.filter_num * len(self.filter_sizes)
        h_pool = tf.concat(pool_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, total_filter_num])
        # Output shape[batch, total_filter_num]

        # Full-connected layer
        # ========================================================================
        with tf.name_scope('dense-%d' % self.dense_unit_num):
            h_full = tf.layers.dense(
                h_pool_flat,
                units=self.dense_unit_num,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1)
            )
            h_full = tf.layers.dropout(h_full, rate=0.5)
            h_full = tf.nn.relu(h_full)
        # =========================================================================

        # Output layer
        with tf.name_scope('output'):
            score = tf.layers.dense(
                h_full,
                units=self.class_num,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1)
            )
            self.score = tf.multiply(score, 1, name='score')
            self.prediction = tf.argmax(score, 1, name='prediction')

        # Loss function
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def prepare_data(self):
        # Data preparation.
        # =======================================================
        # 把预训练词向量的值读到变量中
        self.vecs_dict=np.load('./Data/vocdic.npy').item()
        self.embedding_W = np.load('./Data/vec.npy')
        return
        # =============================================================
        # Date preparation ends.
