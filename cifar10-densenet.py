#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
CIFAR10 DenseNet example. See: http://arxiv.org/abs/1608.06993
Code is developed based on Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
Results using DenseNet (L=40, K=12) on Cifar10 with data augmentation: ~5.77% test error.

Running time:
On one TITAN X GPU (CUDA 7.5 and cudnn 5.1), the code should run ~5iters/s on a batch size 64.
"""

BATCH_SIZE = 64

class Model(ModelDesc):
    def __init__(self, depth):
        super(Model, self).__init__()
        self.N = int((depth - 4)  / 3)
        self.growthRate =12

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 128.0 - 1

        def conv(name, l, channel, stride):
            return Conv2D(name, l, channel, 3, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))
        def add_layer(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                c = BatchNorm('bn1', l)
                c = tf.nn.relu(c)
                c = conv('conv1', c, self.growthRate, 1)
                l = tf.concat([c, l], 3)
            return l

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                l = BatchNorm('bn1', l)
                l = tf.nn.relu(l)
                l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
                l = AvgPooling('pool', l, 2)
            return l


        def dense_net(name):
            l = conv('conv0', image, 16, 1)
            with tf.variable_scope('block1') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition1', l)

            with tf.variable_scope('block2') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition2', l)

            with tf.variable_scope('block3') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
            l = BatchNorm('bnlast', l)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)

            return logits

        logits = dense_net("dense_net")

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            #imgaug.Brightness(20),
            #imgaug.Contrast((0.6,1.4)),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config():
    log_dir = 'train_log/cifar10-single-fisrt%s-second%s-max%s' % (str(args.drop_1), str(args.drop_2), str(args.max_epoch))
    logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)])
        ],
        model=Model(depth=args.depth),
        steps_per_epoch=steps_per_epoch,
        max_epoch=args.max_epoch,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--drop_1',default=150, help='Epoch to drop learning rate to 0.01.') # nargs='*' in multi mode
    parser.add_argument('--drop_2',default=225,help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--depth',default=40, help='The depth of densenet')
    parser.add_argument('--max_epoch',default=300,help='max epoch')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
