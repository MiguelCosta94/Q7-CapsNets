from tensorflow.keras import layers, initializers
import tensorflow.keras.backend as k
import tensorflow as tf
import math


class PrimaryCapsule(layers.Layer):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    def __init__(self, num_capsule, dim_capsule, kernel_size, strides, padding, **kwargs):
        super().__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.output_ns = 0
        self.conv2D = layers.Conv2D(filters=dim_capsule * num_capsule, kernel_size=kernel_size, strides=strides,
                                    padding=padding, name='primarycap_conv2d')

        self.reshape = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')

    def call(self, inputs, training=None):
        self.output_ns = self.conv2D(inputs)
        self.output_ns = self.reshape(self.output_ns)
        output = squash(self.output_ns)

        return output

    def get_output_ns(self):
        return self.output_ns

    def get_config(self):
        config = super().get_config()
        config['num_capsule'] = self.num_capsule
        config['dim_capsule'] = self.dim_capsule
        config['kernel_size'] = self.kernel_size
        config['strides'] = self.strides
        config['padding'] = self.padding
        return config


class Capsule(layers.Layer):
    """
    The capsule layer. It's similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape =
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get('glorot_uniform')
        self.input_hat = 0
        self.output_ns_list = [None] * routings
        self.output_s_list = [None] * routings
        self.cc_list = [None] * routings
        self.b_inst_list = [None] * (routings-1)
        self.b_old_list = [None] * (routings-1)
        self.b_new_list = [None] * (routings-1)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Weight matrix
        self.w = self.add_weight(shape=[self.num_capsule, self.input_num_capsule, self.dim_capsule,
                                        self.input_dim_capsule], initializer=self.kernel_initializer, name='weights')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expanded.shape=[None, 1, input_num_capsule, input_dim_capsule, 1]
        inputs_expanded = k.expand_dims(x=inputs, axis=1)
        inputs_expanded = k.expand_dims(x=inputs_expanded, axis=-1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule, 1]
        inputs_tiled = k.tile(x=inputs_expanded, n=[1, self.num_capsule, 1, 1, 1])

        # Compute `weights * inputs` by scanning weights on dimension 0.
        # x.shape=[None, num_capsule, input_num_capsule, input_dim_capsule, 1]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the 2 inner dimensions as valid matrix multiplication dimensions and outer dimensions as batch size
        # then matmul: [dim_capsule, input_dim_capsule] x [input_dim_capsule, 1] -> [dim_capsule, 1]
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        self.input_hat = k.map_fn(lambda x: tf.linalg.matmul(self.w, x), elems=inputs_tiled)
        self.input_hat = tf.squeeze(input=self.input_hat, axis=-1)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[k.shape(self.input_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.math.scalar_mul(1.0/math.log(2.0), b)
            c = tf.nn.softmax(logits=c, axis=1)
            self.cc_list[i] = c

            # c_expanded.shape =  [None, num_capsule, 1, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmul: [1, input_num_capsule] x [input_num_capsule, dim_capsule] -> [1, dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            c_expanded = k.expand_dims(x=c, axis=-2)
            output_ns = tf.linalg.matmul(a=c_expanded, b=self.input_hat)
            output_ns = tf.squeeze(input=output_ns, axis=-2)
            output = squash(vectors=output_ns)
            self.output_ns_list[i] = output_ns
            self.output_s_list[i] = output

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # outputs_expanded.shape =  [None, num_capsule, dim_capsule, 1]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmul: [input_num_capsule, dim_capsule] x [dim_capsule, 1]
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                output_expanded = k.expand_dims(x=output, axis=-1)
                aux_b = tf.linalg.matmul(a=self.input_hat, b=output_expanded)
                aux_b = tf.squeeze(input=aux_b, axis=-1)
                self.b_inst_list[i] = aux_b
                self.b_old_list[i] = b
                b += aux_b
                self.b_new_list[i] = b
            # End: Routing algorithm -----------------------------------------------------------------------#

        return output

    def set_weights(self, weights):
        self.w = weights

    def get_input_hat(self):
        return self.input_hat

    def get_output_ns_list(self):
        return self.output_ns_list

    def get_output_s_list(self):
        return self.output_s_list

    def get_cc_list(self):
        return self.cc_list

    def get_b_inst_list(self):
        return self.b_inst_list

    def get_b_new_list(self):
        return self.b_new_list

    def get_b_old_list(self):
        return self.b_old_list

    def get_config(self):
        config = super().get_config()
        config['num_capsule'] = self.num_capsule
        config['dim_capsule'] = self.dim_capsule
        config['routings'] = self.routings
        return config


class Length(layers.Layer):
    """
    Compute the length of vectors
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        squared_norm = tf.math.reduce_sum(tf.math.square(inputs), axis=-1)
        norm = tf.math.sqrt(squared_norm)
        return norm

    def get_config(self):
        config = super().get_config()
        return config


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * k.square(k.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * k.square(k.maximum(0., y_pred - 0.1))
    return k.mean(k.sum(L, 1))


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    squared_norm = tf.math.reduce_sum(tf.math.square(vectors), axis=axis, keepdims=True)
    scale = tf.math.sqrt(squared_norm) / (1 + squared_norm)

    return scale * vectors