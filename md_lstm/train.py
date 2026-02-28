import argparse
import logging
from datetime import time
import numpy as np
from enum import Enum
from sys import platform

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import slim

from utils.images import Dataset

plt.rcParams.update({'font.size': 6})
# plt.rcParams['figure.figsize'] = 5, 10

from classes_pascal import pascal_ids
from md_lstm.md_lstm_implementation import multi_dimensional_rnn_while_loop

logger = logging.getLogger(__name__)


def get_script_arguments():
    parser = argparse.ArgumentParser(description='MD LSTM trainer.')
    parser.add_argument('--model_type',
                        required=True,
                        type=ModelType.from_string,
                        choices=list(ModelType), help='Model type.')
    parser.add_argument('--enable_plotting', action='store_true')

    args = get_arguments(parser)
    logger.info('Script inputs: {}.'.format(args))
    return args


class ModelType(Enum):
    MD_LSTM = 'MD_LSTM'

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()


def get_arguments(parser: argparse.ArgumentParser):
    args = None
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        exit(1)
    return args


def train():
    learning_rate = 1e-3
    batch_size = 2
    epochs = 5
    h = 96
    w = 96
    channels = 27  # 3x3x3 patch features from Dataset
    hidden_size = 40
    how_many_classes = 2
    eps = 1e-4
    h_patches = h // 3  # patch grid height
    w_patches = w // 3  # patch grid width

    if platform == 'linux':
        directory_voc_dataset = '/home/pseweryn/Repositories/VOCdevkit/VOC2012'
    else:
        directory_voc_dataset = '/Users/patrykseweryn/PycharmProjects/datasets/voc_dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'

    dataset = Dataset(directory_voc_dataset, 'train', [15], image_shape=(h, w))
    data_X, data_y = dataset.generate_data(500)
    print(len(data_X))
    print(len(data_y))

    x = tf.placeholder(tf.float32, [batch_size, h_patches, w_patches, channels])
    x_v = tf.placeholder(tf.float32, [batch_size, h_patches, w_patches, channels])
    x_h = tf.placeholder(tf.float32, [batch_size, h_patches, w_patches, channels])
    x_vh = tf.placeholder(tf.float32, [batch_size, h_patches, w_patches, channels])

    y = tf.placeholder(tf.float32, [batch_size, h_patches, w_patches, how_many_classes])

    logger.info('Using Multi Dimensional LSTM.')

    rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                  input_data=x, sh=[1, 1],
                                                  scope_n='layer_1')
    rnn_out_v, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                    input_data=x_v, sh=[1, 1],
                                                    scope_n='layer_2')
    rnn_out_h, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                    input_data=x_h, sh=[1, 1],
                                                    scope_n='layer_3')
    rnn_out_vh, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                     input_data=x_vh, sh=[1, 1],
                                                     scope_n='layer_4')

    model_out = slim.fully_connected(
        inputs=tf.concat([rnn_out, rnn_out_v, rnn_out_h, rnn_out_vh], axis=3),
        num_outputs=hidden_size,
        activation_fn=tf.nn.tanh)

    model_out_v = tf.image.flip_left_right(model_out)
    model_out_h = tf.image.flip_up_down(model_out)
    model_out_vh = tf.image.flip_up_down(model_out_v)

    rnn_out_2, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                    input_data=model_out,
                                                    sh=[1, 1],
                                                    scope_n='layer_2_1')

    rnn_out_2_v, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                      input_data=model_out_v,
                                                      sh=[1, 1],
                                                      scope_n='layer_2_2')

    rnn_out_2_h, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                      input_data=model_out_h,
                                                      sh=[1, 1],
                                                      scope_n='layer_2_3')

    rnn_out_2_vh, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                       input_data=model_out_vh,
                                                       sh=[1, 1],
                                                       scope_n='layer_2_4')

    model_logits = slim.fully_connected(
        inputs=tf.concat([rnn_out_2, rnn_out_2_v, rnn_out_2_h, rnn_out_2_vh],
                         axis=3),
        num_outputs=how_many_classes,
        activation_fn=None)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(y, tf.float32), logits=model_logits))
    model_output = tf.nn.softmax(model_logits)
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    # fp = FileLogger('out_{}.tsv'.format(model_type),
    #                 ['steps_{}'.format(model_type),
    #                  'overall_loss_{}'.format(model_type),
    #                  'time_{}'.format(model_type),
    #                  'relevant_loss_{}'.format(model_type)])

    # names = [str(op.name) for op in tf.get_default_graph().get_operations()]

    # data_X shape: (4, N, h_patches, w_patches, 27) - 4 transformations, N images
    # data_y shape: (N, h_patches, w_patches, num_classes)
    num_images = data_X.shape[1]

    for epoch in range(epochs):
        steps = num_images // batch_size
        print('number of steps: {}'.format(steps))
        for i in range(steps):
            idx_start = i * batch_size
            idx_end = (i + 1) * batch_size
            batch_x_orig = data_X[0, idx_start:idx_end] + eps
            batch_x_v = data_X[1, idx_start:idx_end] + eps
            batch_x_h = data_X[2, idx_start:idx_end] + eps
            batch_x_vh = data_X[3, idx_start:idx_end] + eps
            print(np.min(batch_x_orig), np.max(batch_x_orig))
            batch_y = data_y[idx_start:idx_end]

            model_preds, tot_loss_value, _ = sess.run(
                [model_output, loss, grad_update], feed_dict={x: batch_x_orig,
                                                           x_v: batch_x_v,
                                                           x_h: batch_x_h,
                                                           x_vh: batch_x_vh,
                                                           y: batch_y})


            # print('model preds: {}'.format(model_preds.shape))
            print('total_loss_value: {}'.format(tot_loss_value))
            print(model_preds.shape)

            # output_image = np.argmax(model_preds[0], axis=2)
            # print(np.min(output_image), np.max(output_image))
            nrows = 5
            ncols = 8

            if i % 2 == 0:
                # plt.subplot(131)
                # plt.imshow(model_preds[0, :, :, 0], vmin=0, vmax=1)
                # plt.subplot(132)
                # plt.imshow(model_preds[0, :, :, 1], vmin=0, vmax=1)
                # plt.subplot(133)
                # plt.imshow(model_preds[0, :, :, 2], vmin=0, vmax=1)
                plt.subplot(151)
                plt.imshow(batch_x_orig[0, :, :, :3])
                plt.title('oryginał')
                plt.axis('off')

                plt.subplot(152)
                plt.imshow(model_preds[0, :, :, 1], cmap='Greys', vmin=0, vmax=1)
                plt.title('predykcja: człowiek')
                plt.axis('off')

                plt.subplot(153)
                plt.imshow(batch_y[0, :, :, 1], cmap='Greys', vmin=0, vmax=1)
                plt.title('segmentacja: człowiek')
                plt.axis('off')

                plt.subplot(154)
                plt.imshow(model_preds[0, :, :, 0], cmap='Greys', vmin=0, vmax=1)
                plt.title('predykcja: tło')
                plt.axis('off')

                plt.subplot(155)
                plt.imshow(batch_y[0, :, :, 0], cmap='Greys', vmin=0, vmax=1)
                plt.title('segmentacja: tło')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(
                    '/home/pseweryn/Projects/multidimensional_lstm/repository/results/md_lstm/two_step_md_lstm/image_{}_{}.jpg'.format(
                        epoch, i), bbox_inches='tight', dpi=100)
                plt.close()
                # plt.show()

                # plt.savefig(
                #     '/home/pseweryn/Projects/multidimensional_lstm/repository/results/md_lstm/original/original_iteration_{}.jpg'.format(
                #         i))

                # fig, ax = plt.subplots(nrows, ncols)
                # fig.set_size_inches((8, 2), forward=False)
                # for row in range(nrows):
                #     for col in range(ncols):
                #         ax[row, col].imshow(batch_y[, :, :, col + row * ncols], vmin=0, vmax=1)
                #         ax[row, col].set_title(pascal_ids[col + row * ncols])
                #         ax[row, col].axis('off')
                # fig.savefig(
                #     '/home/pseweryn/Projects/multidimensional_lstm/repository/results/md_lstm/two_step_md_lstm/seg_true_iteration_{}.jpg'.format(
                #         i))
                # plt.show()
                # plt.close(fig)

                # fig, ax = plt.subplots(nrows, ncols)
                # fig.set_size_inches((8, 2), forward=False)
                # for row in range(nrows):
                #     for col in range(ncols):
                #         ax[row, col].imshow(model_preds[0, :, :, col + row * ncols], vmin=0, vmax=1)
                #         ax[row, col].set_title(pascal_ids[col + row * ncols])
                #         ax[row, col].axis('off')
                # plt.savefig(
                #     '/home/pseweryn/Projects/multidimensional_lstm/repository/results/md_lstm/segmentation_predicted/seg_predicted_iteration_{}.jpg'.format(
                #         i))
                # plt.show()
                # plt.close(fig)
                # plt.imshow(model_preds[3], vmin=0, vmax=1)
                # plt.imshow(model_preds[4], vmin=0, vmax=1)
                # plt.imshow(model_preds[5], vmin=0, vmax=1)
                # plt.imshow(model_preds[0, :, :, 20])
                # print(np.max(model_preds[0, :, :, 20]))
                # plt.show()

            """
            ____________
            |          |
            |          |
            |     x    |
            |      x <----- extract this prediction. Relevant loss is only computed for this value.
            |__________|    we don't care about the rest (even though the model is trained on all values
                            for simplicity). A standard LSTM should have a very high value for relevant loss
                            whereas a MD LSTM (which can see all the TOP LEFT corner) should perform well. 
            """

            # extract the predictions for the second x
            # relevant_pred_index = get_relevant_prediction_index(batch_y)
            # true_rel = np.array([batch_y[i, x, y, 0] for (i, (y, x)) in
            #                      enumerate(relevant_pred_index)])
            # pred_rel = np.array([model_preds[i, x, y, 0] for (i, (y, x)) in
            #                      enumerate(relevant_pred_index)])
            # relevant_loss = np.mean(np.square(true_rel - pred_rel))
            #
            # values = [str(i).zfill(4), tot_loss_value,
            #           time() - grad_step_start_time, relevant_loss]
            # format_str = 'steps = {0} | overall loss = {1:.3f} | time {2:.3f} | relevant loss = {3:.3f}'
            # logger.info(format_str.format(*values))
            # fp.write(values)

            # display_matplotlib_every = 500
            # if enable_plotting and i % display_matplotlib_every == 0 and i != 0:
            #     visualise_mat(
            #         sess.run(model_out, feed_dict={x: batch_x})[0].squeeze())
            #     visualise_mat(batch_y[0].squeeze())


def main():
    # args = get_script_arguments()
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    train()


if __name__ == '__main__':
    main()
