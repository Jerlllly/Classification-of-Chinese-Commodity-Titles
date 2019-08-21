#coding=UTF-8
import numpy
class DataSet(object):

    def __init__(self,data,labels):
        self._data=data
        self._labels = labels
        self._epochs_completed = 0 # 已经经过了多少个epoch
        self._index_in_epoch = 0 # 在一个epoch中的index
        self._num_examples=len(data) #是指训练数据的样本总个数

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch  # self._index_in_epoch  所有的调用，总共用了多少个样本，相当于一个全局变量 #start第一个batch为0，剩下的就和self._index_in_epoch一样，如果超过了一个epoch，在下面还会重新赋值。
        # Shuffle for the first epoch 第一个epoch需要shuffle
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)  # 生成的一个所有样本长度的np.array
            numpy.random.shuffle(perm0)
            self._data = self._data[perm0]
            self._labels = self._labels[perm0]
        # Go to the next epoch

        if start + batch_size > self._num_examples:  # epoch的结尾和下一个epoch的开头
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start  # 最后不够一个batch还剩下几个
            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._data = self._data[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((data_rest_part, data_new_part), axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:  # 除了第一个epoch，以及每个epoch的开头，剩下中间batch的处理方式
            self._index_in_epoch += batch_size  # start = index_in_epoch
            end = self._index_in_epoch  # end很简单，就是 index_in_epoch加上batch_size
            return self._data[start:end], self._labels[start:end]  # 在数据x,y
