# coding=utf-8
import sklearn.metrics as metrics
import sklearn as sk
import tensorflow as tf
from cnn_model import TextCNN
from cnn_model import CNNConfig
import datetime
import time
import os
import os
from Dataset import *
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"
train_dataset=DataSet(np.load('./Data/TrainData.npy'),np.load('./Data/TrainLabel.npy'))
test_dataset=DataSet(np.load('./Data/TestData.npy'),np.load('./Data/TestLabel.npy'))
def train():
    # Training procedure
    # ======================================================
    # 设定最小显存使用量
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        config = CNNConfig('CHAR-RANDOM')
        cnn = TextCNN(config)
        cnn.prepare_data()
        cnn.setCNN()

        print('Setting Tensorboard and Saver...')
        # 设置Saver和checkpoint来保存模型
        # ===================================================
        checkpoint_dir = os.path.join(os.path.abspath("checkpoints"), "textcnn")
        checkpoint_prefix = os.path.join(checkpoint_dir, cnn.train_mode)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        # =====================================================

        # 配置Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        # ====================================================================
        train_tensorboard_dir = 'tensorboard/textcnn/train/' + config.train_mode
        valid_tensorboard_dir = 'tensorboard/textcnn/valid/' + config.train_mode
        if not os.path.exists(train_tensorboard_dir):
            os.makedirs(train_tensorboard_dir)
        if not os.path.exists(valid_tensorboard_dir):
            os.makedirs(valid_tensorboard_dir)

        # 训练结果记录
        log_file = open(valid_tensorboard_dir+'/log.txt', mode='w')

        merged_summary = tf.summary.merge([tf.summary.scalar('Trainloss', cnn.loss),
                                            tf.summary.scalar('Trainaccuracy', cnn.accuracy)])
        merged_summary_t = tf.summary.merge([tf.summary.scalar('Testloss', cnn.loss),
                                           tf.summary.scalar('Testaccuracy', cnn.accuracy)])
        train_summary_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
        # =========================================================================

        global_step = tf.Variable(0, trainable=False)

        # 保证Batch normalization的执行
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。
            train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(cnn.loss, global_step)

        # 训练步骤
        def train_step(batch_x, batch_y, keep_prob=config.dropout_keep_prob):
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: keep_prob,
                cnn.training: True
            }
            sess.run(train_op, feed_dict=feed_dict)
            step, loss, accuracy, summery = sess.run(
                [global_step, cnn.loss, cnn.accuracy, merged_summary],
                feed_dict={cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: 1.0,
                cnn.training: False})
            t = datetime.datetime.now().strftime('%m-%d %H:%M')
            print('TRAIN  %s: epoch: %d, step: %d, loss: %f, accuracy: %f' % (t, epoch, step, loss, accuracy))
            # 把结果写入Tensorboard中
            train_summary_writer.add_summary(summery, step)

        def test_step(batch_x, batch_y):

            step, loss, accuracy, summery = sess.run(
                [global_step, cnn.loss, cnn.accuracy, merged_summary_t],
                feed_dict={cnn.input_x: batch_x,
                           cnn.labels: batch_y,
                           cnn.dropout_keep_prob: 1.0,
                           cnn.training: False})
            t = datetime.datetime.now().strftime('%m-%d %H:%M')
            print('TEST %s: epoch: %d, step: %d, loss: %f, accuracy: %f' % (t, epoch, step, loss, accuracy))
            # 把结果写入Tensorboard中
            train_summary_writer.add_summary(summery, step)
            return accuracy

        print('Start training TextCNN, training mode='+cnn.train_mode)
        sess.run(tf.global_variables_initializer())

        last=0
        # Training loop
        for epoch in range(1000000):
            batch_x, batch_y=train_dataset.next_batch(128)
            train_step(batch_x, batch_y, config.dropout_keep_prob)
            if epoch%10==0:
                batch_x, batch_y = test_dataset.next_batch(128)
                accuracy=test_step(batch_x, batch_y)
                if accuracy>last:
                    path = saver.save(sess, checkpoint_prefix, global_step=global_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    last=accuracy




        train_summary_writer.close()
        log_file.close()
        # 训练完成后保存参数

    # ==================================================================


if __name__ == '__main__':
    train()