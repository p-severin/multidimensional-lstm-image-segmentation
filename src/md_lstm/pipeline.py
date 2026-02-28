import logging
import os

import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import tf_slim as slim

from md_lstm.md_lstm_implementation import multi_dimensional_rnn_while_loop
from utils.data_generator import DataGenerator
from utils.pascal_voc import resolve_class_names

plt.rcParams.update({'font.size': 6})

logger = logging.getLogger(__name__)

DATA_DIR = 'data/voc_2012_segmentation_data'


def train():
    learning_rate = 1e-3
    batch_size = 4
    epochs = 5
    h = 96
    w = 96
    channels = 27
    hidden_size = 40
    how_many_classes = 2
    eps = 1e-4
    max_grad_norm = 5.0
    class_weights = [0.3, 0.7]
    h_patches = h // 3
    w_patches = w // 3

    chosen_classes = resolve_class_names(['person'])
    dim = (h_patches, w_patches)

    generator = DataGenerator(DATA_DIR, 'train', chosen_classes, batch_size=batch_size, dim=dim, shuffle=True)

    x = tf.placeholder(tf.float32, [batch_size, h_patches, w_patches, channels])
    x_v = tf.placeholder(tf.float32, [batch_size, h_patches, w_patches, channels])
    x_h = tf.placeholder(tf.float32, [batch_size, h_patches, w_patches, channels])
    x_vh = tf.placeholder(tf.float32, [batch_size, h_patches, w_patches, channels])

    y = tf.placeholder(tf.float32, [batch_size, h_patches, w_patches, how_many_classes])

    rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1], scope_n='layer_1')
    rnn_out_v, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x_v, sh=[1, 1], scope_n='layer_2')
    rnn_out_h, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x_h, sh=[1, 1], scope_n='layer_3')
    rnn_out_vh, _ = multi_dimensional_rnn_while_loop(
        rnn_size=hidden_size, input_data=x_vh, sh=[1, 1], scope_n='layer_4'
    )

    model_out = slim.fully_connected(
        inputs=tf.concat([rnn_out, rnn_out_v, rnn_out_h, rnn_out_vh], axis=3),
        num_outputs=hidden_size,
        activation_fn=tf.nn.tanh,
    )

    model_out_v = tf.image.flip_left_right(model_out)
    model_out_h = tf.image.flip_up_down(model_out)
    model_out_vh = tf.image.flip_up_down(model_out_v)

    rnn_out_2, _ = multi_dimensional_rnn_while_loop(
        rnn_size=hidden_size, input_data=model_out, sh=[1, 1], scope_n='layer_2_1'
    )
    rnn_out_2_v, _ = multi_dimensional_rnn_while_loop(
        rnn_size=hidden_size, input_data=model_out_v, sh=[1, 1], scope_n='layer_2_2'
    )
    rnn_out_2_h, _ = multi_dimensional_rnn_while_loop(
        rnn_size=hidden_size, input_data=model_out_h, sh=[1, 1], scope_n='layer_2_3'
    )
    rnn_out_2_vh, _ = multi_dimensional_rnn_while_loop(
        rnn_size=hidden_size, input_data=model_out_vh, sh=[1, 1], scope_n='layer_2_4'
    )

    model_logits = slim.fully_connected(
        inputs=tf.concat([rnn_out_2, rnn_out_2_v, rnn_out_2_h, rnn_out_2_vh], axis=3),
        num_outputs=how_many_classes,
        activation_fn=None,
    )

    weights = tf.constant(class_weights, dtype=tf.float32)
    pixel_weights = tf.reduce_sum(y * weights, axis=-1)
    per_pixel_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(y, tf.float32), logits=model_logits)
    loss = tf.reduce_mean(per_pixel_loss * pixel_weights)

    model_output = tf.nn.softmax(model_logits)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_grads_and_vars = [
        (tf.clip_by_norm(g, max_grad_norm), v) if g is not None else (g, v) for g, v in grads_and_vars
    ]
    grad_update = optimizer.apply_gradients(clipped_grads_and_vars)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        steps = len(generator)
        print(f'Epoch {epoch + 1}/{epochs}, steps: {steps}')

        for step in range(steps):
            batch_X, batch_y = generator[step]

            model_preds, tot_loss_value, _ = sess.run(
                [model_output, loss, grad_update],
                feed_dict={
                    x: batch_X[0] + eps,
                    x_v: batch_X[1] + eps,
                    x_h: batch_X[2] + eps,
                    x_vh: batch_X[3] + eps,
                    y: batch_y,
                },
            )

            print(f'  step {step}/{steps}, loss: {tot_loss_value:.4f}')

            if step % 10 == 0:
                plt.subplot(151)
                plt.imshow(batch_X[0][0, :, :, :3])
                plt.title('original')
                plt.axis('off')

                plt.subplot(152)
                plt.imshow(model_preds[0, :, :, 1], cmap='Greys', vmin=0, vmax=1)
                plt.title('pred: person')
                plt.axis('off')

                plt.subplot(153)
                plt.imshow(batch_y[0, :, :, 1], cmap='Greys', vmin=0, vmax=1)
                plt.title('gt: person')
                plt.axis('off')

                plt.subplot(154)
                plt.imshow(model_preds[0, :, :, 0], cmap='Greys', vmin=0, vmax=1)
                plt.title('pred: bg')
                plt.axis('off')

                plt.subplot(155)
                plt.imshow(batch_y[0, :, :, 0], cmap='Greys', vmin=0, vmax=1)
                plt.title('gt: bg')
                plt.axis('off')

                plt.tight_layout()
                results_dir = 'output/md_lstm'
                os.makedirs(results_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(results_dir, f'image_{epoch}_{step}.jpg'),
                    bbox_inches='tight',
                    dpi=100,
                )
                plt.close()

        generator.on_epoch_end()


def main():
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)
    train()


if __name__ == '__main__':
    main()
