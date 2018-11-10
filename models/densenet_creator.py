import tensorflow as tf

from tensorflow.layers import average_pooling2d, batch_normalization, conv2d, dense, dropout, max_pooling2d
from tensorflow.keras.layers import concatenate, GlobalAveragePooling2D


class DenseNetCreator:
    def __init__(self, img_input, nb_classes, bottleneck=False, data_format='channels_first', depth=40, dropout_rate=0.,
                 efficient=False, growth_rate=12, include_top=True, nb_dense_block=3, nb_filter=-1,
                 nb_layers_per_block=-1, training=True, trainable=True, reduction=0.0, subsample_initial_block=False):
        """ Initialise the DenseNet model creator.

            Args:
                img_input (tensor): Input tensor.
                nb_classes (int): number of classes
                bottleneck (bool, default: False): use bottleneck blocks
                data_format (str, default: 'channels_first'): The dataformat to use for the network
                depth (int, default: 40): number of layers
                dropout_rate (float, default:0.): dropout rate
                efficient (bool, default: False): Whether to run the slower but more memory efficient model or not
                growth_rate (int, default: 12): number of filters to add per dense block
                include_top (bool, default: True): Whether to include a dense classification head.
                nb_dense_block (int, default: 3): number of dense blocks to add to end (generally = 3)
                nb_filter (int, default: -1): initial number of filters. Default -1 indicates initial number of
                                                filters is 2 * growth_rate
                nb_layers_per_block: number of layers in each dense block.
                        Can be a -1, positive integer or a list.
                        If -1, calculates nb_layer_per_block from the depth of the network.
                        If positive integer, a set number of layers per dense block.
                        If list, nb_layer is used as provided. Note that list size must
                        be (nb_dense_block + 1)
                training (bool): Whether it is training or not.
                trainable (bool): Whether to add the vairbales to tf.GraphKeys.TRAINABLE_VARIABLES
                reduction (float, default: 0.): reduction factor of transition blocks. Note : reduction value is
                                                inverted to compute compression
                subsample_initial_block (bool, defualt: False): Set to True to subsample the initial convolution and
                        add a MaxPool2D before the dense blocks are added.

        """
        self.axis = 1 if data_format == 'channels_first' else 3

        self.bottleneck = bottleneck
        self.bn_kwargs = {'fused': True,
                          'axis': self.axis,
                          'training': training,
                          'trainable': trainable}

        self.conv_kwargs = {'data_format': data_format, 'trainable': trainable}

        self.data_format = data_format
        self.depth = depth
        self.dropout_rate = dropout_rate

        self.efficient = efficient

        self.growth_rate = growth_rate

        self.img_input = img_input
        self.include_top = include_top

        self.nb_classes = nb_classes
        self.nb_dense_block = nb_dense_block
        self.nb_filter = nb_filter

        self.subsample_initial_block = subsample_initial_block

        self.training = training
        self.trainable = trainable

        if reduction != 0.0:
            assert 0 < reduction <= 1.0, 'reduction value must lie between 0.0 and 1.0'

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            assert len(nb_layers) == nb_dense_block, 'If list, nb_layer is used as provided. ' \
                                                     'Note that list size must be (nb_dense_block)'
            self.final_nb_layer = nb_layers[-1]
            self.nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
                count = int((depth - 4) / 3)
                self.nb_layers = [count for _ in range(nb_dense_block)]
                self.final_nb_layer = count
            else:
                self.final_nb_layer = nb_layers_per_block
                self.nb_layers = [nb_layers_per_block] * nb_dense_block

        # compute initial nb_filter if -1, else accept users initial nb_filter
        if self.nb_filter <= 0:
            self.nb_filter = 2 * self.growth_rate

        # compute compression factor
        self.compression = 1.0 - reduction

        # Initial convolution
        if self.subsample_initial_block:
            self.initial_kernel = (7, 7)
            self.initial_strides = (2, 2)
        else:
            self.initial_kernel = (3, 3)
            self.initial_strides = (1, 1)

    def _conv_block(self, ip, nb_filter):
        """ Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout

            Args:
                ip: Input tensor
                nb_filter: number of filters

            Returns: tensor with batch_norm, relu and convolution2d added (optional bottleneck)
        """

        def _x(ip):
            x = batch_normalization(ip, **self.bn_kwargs)
            x = tf.nn.relu(x)

            if self.bottleneck:
                inter_channel = nb_filter * 4

                x = conv2d(x, inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                           **self.conv_kwargs)
                x = batch_normalization(x, **self.bn_kwargs)
                x = tf.nn.relu(x)

            x = conv2d(x, nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False,
                       **self.conv_kwargs)

            if self.dropout_rate:
                x = dropout(x, self.dropout_rate, training=self.training)

            return x

        if self.efficient:
            # Gradient checkpoint the layer
            _x = tf.contrib.layers.recompute_grad(_x)

        return _x(ip)

    def _dense_block(self, x, nb_layers, nb_filter, grow_nb_filters=True, return_concat_list=False):
        """ Build a dense_block where the output of each conv_block is fed to subsequent ones

            Args:
                x: tensor
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                grow_nb_filters: flag to decide to allow number of filters to grow
                return_concat_list: return the list of feature maps along with the actual output

            Returns:
                 tensor with nb_layers of conv_block appended
        """
        x_list = [x]

        for i in range(nb_layers):
            with tf.variable_scope('denselayer_{}'.format(i), use_resource=True):
                cb = self._conv_block(x, self.growth_rate)
                x_list.append(cb)

                x = concatenate([x, cb], self.axis)

                if grow_nb_filters:
                    nb_filter += self.growth_rate

        if self.dropout_rate:
            x = dropout(x, self.dropout_rate, training=self.training)

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter

    def _transition_block(self, ip, nb_filter):
        """ Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D

            Args:
                ip: tensor
                nb_filter: number of filters
                compression: calculated as 1 - reduction. Reduces the number of feature maps
                            in the transition block.
                dropout_rate: dropout rate
                weight_decay: weight decay factor

            Returns:
                 tensor, after applying batch_norm, relu-conv, dropout, maxpool
        """
        x = batch_normalization(ip, **self.bn_kwargs)
        x = tf.nn.relu(x)
        x = conv2d(x, int(nb_filter * self.compression), (1, 1), kernel_initializer='he_normal',
                   padding='same', use_bias=False, **self.conv_kwargs)
        x = average_pooling2d(x, (2, 2), strides=(2, 2), data_format=self.data_format)

        return x

    def __call__(self):
        """ Builds the network. """
        x = conv2d(self.img_input, self.nb_filter, self.initial_kernel, kernel_initializer='he_normal', padding='same',
                   strides=self.initial_strides, use_bias=False, **self.conv_kwargs)

        if self.subsample_initial_block:
            x = batch_normalization(x, **self.bn_kwargs)
            x = tf.nn.relu(x)
            x = max_pooling2d(x, (3, 3), data_format=self.data_format, strides=(2, 2), padding='same')

        # Add dense blocks
        nb_filter = self.nb_filter
        for block_idx in range(self.nb_dense_block - 1):
            with tf.variable_scope('denseblock_{}'.format(block_idx)):
                x, nb_filter = self._dense_block(x, self.nb_layers[block_idx], nb_filter)
                # add transition_block
                x = self._transition_block(x, nb_filter)
                nb_filter = int(nb_filter * self.compression)

        # The last dense_block does not have a transition_block
        x, nb_filter = self._dense_block(x, self.final_nb_layer, self.nb_filter)

        x = batch_normalization(x, **self.bn_kwargs)
        x = tf.nn.relu(x)

        x = GlobalAveragePooling2D(data_format=self.data_format)(x)

        if self.include_top:
            x = dense(x, self.nb_classes)

        return x
