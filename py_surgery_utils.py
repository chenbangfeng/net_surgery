###############################################################################
# Common needed utilities for net surgery
# author : Mahdyar Ravanbaksh
###############################################################################

import caffe
import numpy as np


def compute_dist(x_mat, y_mat):
    """ Compute distance between two numpy 2-D array
    :param x_mat: numpy array [N x M]
    :param y_mat: numpy array [N x M]
    :return: numpy array distance between X,Y. dist [N x 1]
    """
    dist_list = []
    for x, y in zip(x_mat, y_mat):
        # noinspection PyTypeChecker
        dist_list.append(np.sqrt(np.sum((x - y) ** 2)))
    return np.array(dist_list)


def forward_nets(image, net, net_full_conv):
    """
    forward same image in two nets and print classification results
    :param image: string image address
    :param net: caffe net first net (standard net)
    :param net_full_conv: caffe net second (modified net)
    :returns out_org, out_conv: network outputs
    """
    # load input and configure pre-processing
    im = caffe.io.load_image(image)
    transformer = caffe.io.Transformer(
            {'data': net_full_conv.blobs['data'].data.shape})
    transformer.set_mean('data',
                         np.load('ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    # classification map; forward and print prediction indices at each location
    out_org = net.forward_all(
            data=np.asarray([transformer.preprocess('data', im)]))
    print '(original net) cat : ', out_org['prob'][0].argmax(axis=0), \
        ' , prob  : ', out_org['prob'][0].max(axis=0)

    out_conv = net_full_conv.forward_all(
            data=np.asarray([transformer.preprocess('data', im)]))
    print '(full_conv net) cat : ', out_conv['prob'][0].argmax(axis=0), \
        ' , prob  : ', out_conv['prob'][0].max(axis=0)

    return out_org, out_conv
