import tensorflow as tf


class MultiDimensionalLSTMCell(tf.keras.layers.Layer):
    """MD-LSTM cell with layer normalization.

    Takes input from two spatial neighbors (above and left), each contributing
    their own cell state and hidden state. Uses separate forget gates for each
    neighbor's cell state.
    """

    def __init__(self, num_units, forget_bias=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.forget_bias = forget_bias

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_dim + 2 * self.num_units, 5 * self.num_units],
            initializer='glorot_uniform',
        )
        ln_shape = [None, self.num_units]
        self.ln_i = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln_i.build(ln_shape)
        self.ln_j = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln_j.build(ln_shape)
        self.ln_f1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln_f1.build(ln_shape)
        self.ln_f2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln_f2.build(ln_shape)
        self.ln_o = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln_o.build(ln_shape)
        self.ln_c = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln_c.build(ln_shape)
        super().build(input_shape)

    def call(self, inputs, c_up, c_left, h_up, h_left):
        concat = tf.matmul(tf.concat([inputs, h_up, h_left], axis=1), self.kernel)
        i, j, f1, f2, o = tf.split(concat, 5, axis=1)

        i = self.ln_i(i)
        j = self.ln_j(j)
        f1 = self.ln_f1(f1)
        f2 = self.ln_f2(f2)
        o = self.ln_o(o)

        new_c = (
            c_up * tf.nn.sigmoid(f1 + self.forget_bias)
            + c_left * tf.nn.sigmoid(f2 + self.forget_bias)
            + tf.nn.sigmoid(i) * tf.nn.tanh(j)
        )
        new_h = tf.nn.tanh(self.ln_c(new_c)) * tf.nn.sigmoid(o)
        return new_h, new_c, new_h

    def get_config(self):
        config = super().get_config()
        config.update({'num_units': self.num_units, 'forget_bias': self.forget_bias})
        return config


class MultiDimensionalLSTM(tf.keras.layers.Layer):
    """Processes a 2D feature map with MD-LSTM in raster-scan order.

    Each spatial position depends on its left and upper neighbors.
    Input: (batch, h, w, channels) → Output: (batch, h, w, rnn_size)
    """

    def __init__(self, rnn_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn_size = rnn_size

    def build(self, input_shape):
        self._h = input_shape[1]
        self._w = input_shape[2]
        channels = input_shape[3]
        self.cell = MultiDimensionalLSTMCell(self.rnn_size, name='cell')
        self.cell.build(tf.TensorShape([None, channels]))
        super().build(input_shape)

    def call(self, input_data):
        batch_size = tf.shape(input_data)[0]
        h, w = self._h, self._w
        rnn_size = self.rnn_size
        hw = h * w

        x = tf.transpose(input_data, [1, 2, 0, 3])
        x = tf.reshape(x, [hw, batch_size, -1])

        inputs_ta = tf.TensorArray(dtype=tf.float32, size=hw)
        inputs_ta = inputs_ta.unstack(x)

        c_ta = tf.TensorArray(dtype=tf.float32, size=hw + 1, clear_after_read=False)
        h_ta = tf.TensorArray(dtype=tf.float32, size=hw + 1, clear_after_read=False)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=hw)

        zero_state = tf.zeros([batch_size, rnn_size])
        c_ta = c_ta.write(hw, zero_state)
        h_ta = h_ta.write(hw, zero_state)

        def body(time, outputs_ta, c_ta, h_ta):
            up_idx = tf.cond(tf.less(time, w), lambda: hw, lambda: time - w)
            left_idx = tf.cond(tf.greater(tf.math.floormod(time, w), 0), lambda: time - 1, lambda: hw)

            out, new_c, new_h = self.cell.call(
                inputs_ta.read(time),
                c_ta.read(up_idx), c_ta.read(left_idx),
                h_ta.read(up_idx), h_ta.read(left_idx),
            )
            outputs_ta = outputs_ta.write(time, out)
            c_ta = c_ta.write(time, new_c)
            h_ta = h_ta.write(time, new_h)
            return time + 1, outputs_ta, c_ta, h_ta

        def condition(time, *_):
            return tf.less(time, hw)

        _, outputs_ta, _, _ = tf.while_loop(
            condition, body,
            [tf.constant(0), outputs_ta, c_ta, h_ta],
            parallel_iterations=1,
        )

        outputs = outputs_ta.stack()
        y = tf.reshape(outputs, [h, w, batch_size, rnn_size])
        y = tf.transpose(y, [2, 0, 1, 3])
        return y

    def get_config(self):
        config = super().get_config()
        config.update({'rnn_size': self.rnn_size})
        return config
