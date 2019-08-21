# coding=utf-8
import sklearn as sk
import sklearn.metrics as metrics
import tensorflow as tf
from cnn_model import TextCNN
from cnn_model import CNNConfig
import os
import datetime
import numpy as np



def predict():
    """
    读取模型，预测商品标题
    :param titles: 列表，商品标题的字符串
    :return: results
    """
    # Test procedure
    # ======================================================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # TODO: 读取不同模型，修改此处参数
        # 要读取的模型路径
        checkpoint_dir = os.path.abspath("checkpoints/textcnn")
        # 模型的文件名放在这，不含后缀
        checkpoint_file = os.path.join(checkpoint_dir, "CHAR-RANDOM-25871")
        # 这要加.meta后缀
        saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, 'CHAR-RANDOM-25871.meta'))
        saver.restore(sess, checkpoint_file)
        graph = tf.get_default_graph()

        # 这里的train_mode参数要和模型一致
        config = CNNConfig('CHAR-RANDOM')
        cnn = TextCNN(config)

        # 从图中读取变量
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        prediction = graph.get_operation_by_name("output/prediction").outputs[0]
        training = graph.get_operation_by_name("training").outputs[0]
        batch_x=np.load('./Data/Data.npy')[0:10000]
        feed_dict = {
            input_x: batch_x,
            dropout_keep_prob: 1.0,
            training: False
        }
        pre = sess.run(prediction, feed_dict)
        return pre




if __name__ == '__main__':
    print(predict())
