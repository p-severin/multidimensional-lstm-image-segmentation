import tensorflow as tf

if int(tf.__version__.split('.')[0]) >= 2:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    from tensorflow.compat.v1.nn.rnn_cell import LSTMStateTuple, RNNCell

    def _linear(args, output_size, bias, bias_start=0.0, scope=None):
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError(f'Linear is expecting 2D arguments: {shapes!s}')
            if not shape[1]:
                raise ValueError(f'Linear expects shape[1] of arguments: {shapes!s}')
            else:
                total_arg_size += shape[1]

        with tf.variable_scope(scope or 'Linear'):
            matrix = tf.get_variable('Matrix', [total_arg_size, output_size])
            if len(args) == 1:
                res = tf.matmul(args[0], matrix)
            else:
                res = tf.matmul(tf.concat(axis=1, values=args), matrix)
            if not bias:
                return res
            bias_term = tf.get_variable('Bias', [output_size], initializer=tf.constant_initializer(bias_start))
        return res + bias_term

else:
    from tensorflow.contrib.rnn import LSTMStateTuple, RNNCell
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear


def ln(tensor, scope=None, epsilon=1e-5):
    """Layer normalizes a 2D tensor along its second axis."""
    assert len(tensor.get_shape()) == 2
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale', shape=[tensor.get_shape()[1]], initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift', shape=[tensor.get_shape()[1]], initializer=tf.constant_initializer(0))
    ln_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return ln_initial * scale + shift


class MultiDimensionalLSTMCell(RNNCell):
    """Adapted from TF's BasicLSTMCell to use Layer Normalization. state_is_tuple is always True."""

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """
        @param inputs: (batch, n)
        @param state: the states and hidden unit of the two cells
        """
        with tf.variable_scope(scope or type(self).__name__):
            c1, c2, h1, h2 = state

            # bias=False because LN adds bias via shift
            concat = _linear([inputs, h1, h2], 5 * self._num_units, False)

            i, j, f1, f2, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            i = ln(i, scope='i/')
            j = ln(j, scope='j/')
            f1 = ln(f1, scope='f1/')
            f2 = ln(f2, scope='f2/')
            o = ln(o, scope='o/')

            new_c = (
                c1 * tf.nn.sigmoid(f1 + self._forget_bias)
                + c2 * tf.nn.sigmoid(f2 + self._forget_bias)
                + tf.nn.sigmoid(i) * self._activation(j)
            )

            new_h = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
            new_state = LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def multi_dimensional_rnn_while_loop(rnn_size, input_data, sh, dims=None, scope_n='layer1'):
    """Implements naive multi dimension recurrent neural networks.

    @param rnn_size: the hidden units
    @param input_data: the data to process of shape [batch,h,w,channels]
    @param sh: [height,width] of the windows
    @param dims: dimensions to reverse the input data, e.g.
        dims=[False,True,True,False] => true means reverse dimension
    @param scope_n: the scope

    returns [batch,h/sh[0],w/sh[1],rnn_size] the output of the lstm
    """

    with tf.variable_scope(scope_n):
        cell = MultiDimensionalLSTMCell(rnn_size)

        shape = input_data.get_shape().as_list()
        batch_size = shape[0]
        X_dim = shape[1]
        Y_dim = shape[2]
        channels = shape[3]
        X_win = sh[0]
        Y_win = sh[1]
        batch_size_runtime = tf.shape(input_data)[0]

        # Pad with zeros if input can't be exactly sampled by the window
        if X_dim % X_win != 0:
            offset = tf.zeros([batch_size_runtime, X_win - (X_dim % X_win), Y_dim, channels])
            input_data = tf.concat(axis=1, values=[input_data, offset])
            shape = input_data.get_shape().as_list()
            X_dim = shape[1]

        if Y_dim % Y_win != 0:
            offset = tf.zeros([batch_size_runtime, X_dim, Y_win - (Y_dim % Y_win), channels])
            input_data = tf.concat(axis=2, values=[input_data, offset])
            shape = input_data.get_shape().as_list()
            Y_dim = shape[2]

        h, w = int(X_dim / X_win), int(Y_dim / Y_win)
        features = Y_win * X_win * channels

        # (batch, h, w, features)
        x = tf.reshape(input_data, [batch_size_runtime, h, w, features])

        if dims is not None:
            assert dims[0] is False and dims[3] is False
            x = tf.reverse(x, dims)

        # (h, w, batch, features)
        x = tf.transpose(x, [1, 2, 0, 3])
        # (h*w*batch, features)
        x = tf.reshape(x, [-1, features])
        # h*w tensors of (batch, features)
        x = tf.split(axis=0, num_or_size_splits=h * w, value=x)

        inputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='input_ta')
        inputs_ta = inputs_ta.unstack(x)
        states_ta = tf.TensorArray(dtype=tf.float32, size=h * w + 1, name='state_ta', clear_after_read=False)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='output_ta')

        # Initial zero states at position h*w
        states_ta = states_ta.write(
            h * w,
            LSTMStateTuple(
                tf.zeros([batch_size_runtime, rnn_size], tf.float32),
                tf.zeros([batch_size_runtime, rnn_size], tf.float32),
            ),
        )

        def get_up(t_, w_):
            return t_ - tf.constant(w_)

        def get_last(t_, w_):
            return t_ - tf.constant(1)

        time = tf.constant(0)
        zero = tf.constant(0)

        def body(time_, outputs_ta_, states_ta_):
            # First row reads zero state, otherwise reads state from row above
            state_up = tf.cond(
                tf.less(time_, tf.constant(w)),
                lambda: states_ta_.read(h * w),
                lambda: states_ta_.read(get_up(time_, w)),
            )

            # First column reads zero state, otherwise reads previous state
            state_last = tf.cond(
                tf.less(zero, tf.mod(time_, tf.constant(w))),
                lambda: states_ta_.read(get_last(time_, w)),
                lambda: states_ta_.read(h * w),
            )

            current_state = state_up[0], state_last[0], state_up[1], state_last[1]
            out, state = cell(inputs_ta.read(time_), current_state)
            outputs_ta_ = outputs_ta_.write(time_, out)
            states_ta_ = states_ta_.write(time_, state)

            return time_ + 1, outputs_ta_, states_ta_

        def condition(time_, outputs_ta_, states_ta_):
            return tf.less(time_, tf.constant(h * w))

        _result, outputs_ta, states_ta = tf.while_loop(
            condition, body, [time, outputs_ta, states_ta], parallel_iterations=1
        )

        outputs = outputs_ta.stack()
        states = states_ta.stack()

        y = tf.reshape(outputs, [h, w, batch_size_runtime, rnn_size])
        y = tf.transpose(y, [2, 0, 1, 3])
        if dims is not None:
            y = tf.reverse(y, dims)

        return y, states
