"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for learners

@author: Konstantin Krismer
"""
from ast import literal_eval

import tensorflow as tf


class KerasHelper:
    @staticmethod
    def get_keras_layer(operation):
        if "input_shape" in operation.parameters:
            input_shape = literal_eval(
                operation.parameters["input_shape"].strip())
        else:
            input_shape = None

        name = operation.name.strip().lower()
        if name == "flatten":
            if input_shape is None:
                return(tf.keras.layers.Flatten())
            else:
                return(tf.keras.layers.Flatten(input_shape=input_shape))
        elif name == "reshape":
            target_shape = literal_eval(operation.parameters["target_shape"].strip())
            if input_shape is None:
                return(tf.keras.layers.Reshape(target_shape = target_shape))
            else:
                return(tf.keras.layers.Reshape(target_shape = target_shape,
                                               input_shape=input_shape))
        elif name == "dense":
            units = int(operation.parameters["units"].strip())

            if "activation" in operation.parameters:
                activation = operation.parameters["activation"].strip()
            else:
                activation = None

            if "use_bias" in operation.parameters:
                use_bias = bool(operation.parameters["use_bias"].strip())
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = \
                    eval(operation.parameters["kernel_initializer"].strip())
            else:
                kernel_initializer = "glorot_uniform"

            if "bias_initializer" in operation.parameters:
                bias_initializer = \
                    eval(operation.parameters["bias_initializer"].strip())
            else:
                bias_initializer = "zeros"

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = \
                    eval(operation.parameters["kernel_regularizer"].strip())
            else:
                kernel_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = \
                    eval(operation.parameters["bias_regularizer"].strip())
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = \
                    eval(operation.parameters["activity_regularizer"].strip())
            else:
                activity_regularizer = None

            if input_shape is None:
                return(tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer
                ))
            else:
                return(tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    input_shape=input_shape
                ))
        elif name == "lstm":
            units = int(operation.parameters["units"].strip())

            if "activation" in operation.parameters:
                activation = operation.parameters["activation"].strip()
            else:
                activation = "tanh"

            if "recurrent_activation" in operation.parameters:
                recurrent_activation = \
                    operation.parameters["recurrent_activation"].strip()
            else:
                recurrent_activation = "sigmoid"

            if "use_bias" in operation.parameters:
                use_bias = bool(operation.parameters["use_bias"].strip())
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = \
                    eval(operation.parameters["kernel_initializer"].strip())
            else:
                kernel_initializer = "glorot_uniform"

            if "recurrent_initializer" in operation.parameters:
                recurrent_initializer = \
                    eval(operation.parameters["recurrent_initializer"].strip())
            else:
                recurrent_initializer = "orthogonal"

            if "bias_initializer" in operation.parameters:
                bias_initializer = \
                    eval(operation.parameters["bias_initializer"].strip())
            else:
                bias_initializer = "zeros"

            if "unit_forget_bias" in operation.parameters:
                unit_forget_bias = \
                    bool(operation.parameters["unit_forget_bias"].strip())
            else:
                unit_forget_bias = True

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = \
                    eval(operation.parameters["kernel_regularizer"].strip())
            else:
                kernel_regularizer = None

            if "recurrent_regularizer" in operation.parameters:
                recurrent_regularizer = \
                    eval(operation.parameters["recurrent_regularizer"].strip())
            else:
                recurrent_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = \
                    eval(operation.parameters["bias_regularizer"].strip())
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = \
                    eval(operation.parameters["activity_regularizer"].strip())
            else:
                activity_regularizer = None

            if "dropout" in operation.parameters:
                dropout = float(operation.parameters["dropout"].strip())
            else:
                dropout = 0.0

            if "recurrent_dropout" in operation.parameters:
                recurrent_dropout = \
                    float(operation.parameters["recurrent_dropout"].strip())
            else:
                recurrent_dropout = 0.0

            if "implementation" in operation.parameters:
                implementation = \
                    int(operation.parameters["implementation"].strip())
            else:
                implementation = 2

            if "return_sequences" in operation.parameters:
                return_sequences = \
                    bool(operation.parameters["return_sequences"].strip())
            else:
                return_sequences = False

            if "return_state" in operation.parameters:
                return_state = \
                    bool(operation.parameters["return_state"].strip())
            else:
                return_state = False

            if "go_backwards" in operation.parameters:
                go_backwards = \
                    bool(operation.parameters["go_backwards"].strip())
            else:
                go_backwards = False

            if "stateful" in operation.parameters:
                stateful = bool(operation.parameters["stateful"].strip())
            else:
                stateful = False

            if "time_major" in operation.parameters:
                time_major = bool(operation.parameters["time_major"].strip())
            else:
                time_major = False

            if "unroll" in operation.parameters:
                unroll = bool(operation.parameters["unroll"].strip())
            else:
                unroll = False

            if input_shape is None:
                return(tf.keras.layers.LSTM(
                    units,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    unit_forget_bias=unit_forget_bias,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    implementation=implementation,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    go_backwards=go_backwards,
                    stateful=stateful,
                    time_major=time_major,
                    unroll=unroll
                ))
            else:
                return(tf.keras.layers.LSTM(
                    units,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    unit_forget_bias=unit_forget_bias,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    implementation=implementation,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    go_backwards=go_backwards,
                    stateful=stateful,
                    time_major=time_major,
                    unroll=unroll,
                    input_shape=input_shape
                ))
        elif name == "conv1d":
            filters = int(operation.parameters["filters"].strip())
            kernel_size = \
                literal_eval(operation.parameters["kernel_size"].strip())

            if "strides" in operation.parameters:
                strides = literal_eval(operation.parameters["strides"].strip())
            else:
                strides = 1

            if "padding" in operation.parameters:
                padding = operation.parameters["padding"].strip()
            else:
                padding = "valid"

            if "data_format" in operation.parameters:
                data_format = operation.parameters["data_format"].strip()
            else:
                data_format = "channels_last"

            if "dilation_rate" in operation.parameters:
                dilation_rate = operation.parameters["dilation_rate"].strip()
            else:
                dilation_rate = 1

            if "activation" in operation.parameters:
                activation = operation.parameters["activation"].strip()
            else:
                activation = None

            if "use_bias" in operation.parameters:
                use_bias = bool(operation.parameters["use_bias"].strip())
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = \
                    eval(operation.parameters["kernel_initializer"].strip())
            else:
                kernel_initializer = "glorot_uniform"

            if "bias_initializer" in operation.parameters:
                bias_initializer = \
                    eval(operation.parameters["bias_initializer"].strip())
            else:
                bias_initializer = "zeros"

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = \
                    eval(operation.parameters["kernel_regularizer"].strip())
            else:
                kernel_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = \
                    eval(operation.parameters["bias_regularizer"].strip())
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = \
                    eval(operation.parameters["activity_regularizer"].strip())
            else:
                activity_regularizer = None

            if input_shape is None:
                return(tf.keras.layers.Conv1D(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer
                ))
            else:
                return(tf.keras.layers.Conv1D(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    input_shape=input_shape
                ))
        elif name == "conv2d":
            filters = int(operation.parameters["filters"].strip())
            kernel_size = \
                literal_eval(operation.parameters["kernel_size"].strip())

            if "strides" in operation.parameters:
                strides = literal_eval(operation.parameters["strides"].strip())
            else:
                strides = (1, 1)

            if "padding" in operation.parameters:
                padding = operation.parameters["padding"].strip()
            else:
                padding = "valid"

            if "data_format" in operation.parameters:
                data_format = operation.parameters["data_format"].strip()
            else:
                data_format = None

            if "dilation_rate" in operation.parameters:
                dilation_rate = operation.parameters["dilation_rate"].strip()
            else:
                dilation_rate = (1, 1)

            if "activation" in operation.parameters:
                activation = operation.parameters["activation"].strip()
            else:
                activation = None

            if "use_bias" in operation.parameters:
                use_bias = bool(operation.parameters["use_bias"].strip())
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = \
                    eval(operation.parameters["kernel_initializer"].strip())
            else:
                kernel_initializer = "glorot_uniform"

            if "bias_initializer" in operation.parameters:
                bias_initializer = \
                    eval(operation.parameters["bias_initializer"].strip())
            else:
                bias_initializer = "zeros"

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = \
                    eval(operation.parameters["kernel_regularizer"].strip())
            else:
                kernel_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = \
                    eval(operation.parameters["bias_regularizer"].strip())
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = \
                    eval(operation.parameters["activity_regularizer"].strip())
            else:
                activity_regularizer = None

            if input_shape is None:
                return(tf.keras.layers.Conv2D(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer
                ))
            else:
                return(tf.keras.layers.Conv2D(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    input_shape=input_shape
                ))
        elif name == "globalmaxpool1d":
            return(tf.keras.layers.GlobalMaxPool1D())

    @staticmethod
    def get_optimizer(optimizer_hyperparameters):
        if "optimizer" in optimizer_hyperparameters:
            optimizer = \
                optimizer_hyperparameters["optimizer"].lower().strip()
            
            if "learning_rate" in optimizer_hyperparameters:
                learning_rate = float(
                    optimizer_hyperparameters["learning_rate"].strip())
            else:
                learning_rate = 0.001

            if "beta_1" in optimizer_hyperparameters:
                beta_1 = float(
                    optimizer_hyperparameters["beta_1"].strip())
            else:
                beta_1 = 0.9

            if "beta_2" in optimizer_hyperparameters:
                beta_2 = float(
                    optimizer_hyperparameters["beta_2"].strip())
            else:
                beta_2 = 0.999

            if "epsilon" in optimizer_hyperparameters:
                epsilon = float(
                    optimizer_hyperparameters["epsilon"].strip())
            else:
                epsilon = 1e-07

            if "clipnorm" in optimizer_hyperparameters:
                clipnorm = float(
                    optimizer_hyperparameters["clipnorm"].strip())
            else:
                clipnorm = None

            if "clipvalue" in optimizer_hyperparameters:
                clipvalue = float(
                    optimizer_hyperparameters["clipvalue"].strip())
            else:
                clipvalue = None

            if optimizer == "adadelta":
                if "rho" in optimizer_hyperparameters:
                    rho = float(optimizer_hyperparameters["rho"].strip())
                else:
                    rho = 0.95
                
                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Adadelta(
                        learning_rate=learning_rate,
                        rho=rho,
                        epsilon=epsilon)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Adadelta(
                        learning_rate=learning_rate,
                        rho=rho,
                        epsilon=epsilon,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Adadelta(
                        learning_rate=learning_rate,
                        rho=rho,
                        epsilon=epsilon,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Adadelta(
                        learning_rate=learning_rate,
                        rho=rho,
                        epsilon=epsilon,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "adagrad":
                if "initial_accumulator_value" in optimizer_hyperparameters:
                    initial_accumulator_value = float(
                        optimizer_hyperparameters["initial_accumulator_value"].strip())
                else:
                    initial_accumulator_value = 0.1
                
                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Adagrad(
                        learning_rate=learning_rate,
                        initial_accumulator_value=initial_accumulator_value,
                        epsilon=epsilon)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Adagrad(
                        learning_rate=learning_rate,
                        initial_accumulator_value=initial_accumulator_value,
                        epsilon=epsilon,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Adagrad(
                        learning_rate=learning_rate,
                        initial_accumulator_value=initial_accumulator_value,
                        epsilon=epsilon,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Adagrad(
                        learning_rate=learning_rate,
                        initial_accumulator_value=initial_accumulator_value,
                        epsilon=epsilon,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "adam":
                if "amsgrad" in optimizer_hyperparameters:
                    amsgrad = bool(
                        optimizer_hyperparameters["amsgrad"].strip())
                else:
                    amsgrad = False
                
                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Adam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        amsgrad=amsgrad)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Adam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        amsgrad=amsgrad,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Adam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        amsgrad=amsgrad,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Adam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        amsgrad=amsgrad,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "adamax":
                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Adamax(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Adamax(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Adamax(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Adamax(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "ftrl":
                if "learning_rate_power" in optimizer_hyperparameters:
                    learning_rate_power = float(
                        optimizer_hyperparameters["learning_rate_power"].strip())
                else:
                    learning_rate_power = -0.5

                if "initial_accumulator_value" in optimizer_hyperparameters:
                    initial_accumulator_value = float(
                        optimizer_hyperparameters["initial_accumulator_value"].strip())
                else:
                    initial_accumulator_value = 0.1

                if "l1_regularization_strength" in optimizer_hyperparameters:
                    l1_regularization_strength = float(
                        optimizer_hyperparameters["l1_regularization_strength"].strip())
                else:
                    l1_regularization_strength = 0.0

                if "l2_regularization_strength" in optimizer_hyperparameters:
                    l2_regularization_strength = float(
                        optimizer_hyperparameters["l2_regularization_strength"].strip())
                else:
                    l2_regularization_strength = 0.0

                if "l2_shrinkage_regularization_strength" in optimizer_hyperparameters:
                    l2_shrinkage_regularization_strength = float(
                        optimizer_hyperparameters["l2_shrinkage_regularization_strength"].strip())
                else:
                    l2_shrinkage_regularization_strength = 0.0
                
                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Ftrl(
                        learning_rate=learning_rate,
                        learning_rate_power=learning_rate_power,
                        initial_accumulator_value=initial_accumulator_value,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Ftrl(
                        learning_rate=learning_rate,
                        learning_rate_power=learning_rate_power,
                        initial_accumulator_value=initial_accumulator_value,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Ftrl(
                        learning_rate=learning_rate,
                        learning_rate_power=learning_rate_power,
                        initial_accumulator_value=initial_accumulator_value,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Ftrl(
                        learning_rate=learning_rate,
                        learning_rate_power=learning_rate_power,
                        initial_accumulator_value=initial_accumulator_value,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "nadam":
                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Nadam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Nadam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Nadam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Nadam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "rmsprop":
                if "rho" in optimizer_hyperparameters:
                    rho = float(optimizer_hyperparameters["rho"].strip())
                else:
                    rho = 0.9

                if "momentum" in optimizer_hyperparameters:
                    momentum = float(
                        optimizer_hyperparameters["momentum"].strip())
                else:
                    momentum = 0.0

                if "centered" in optimizer_hyperparameters:
                    centered = bool(
                        optimizer_hyperparameters["centered"].strip())
                else:
                    centered = False
                
                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.RMSprop(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        epsilon=epsilon,
                        centered=centered)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.RMSprop(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        epsilon=epsilon,
                        centered=centered,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.RMSprop(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        epsilon=epsilon,
                        centered=centered,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.RMSprop(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        epsilon=epsilon,
                        centered=centered,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "sgd":
                if "momentum" in optimizer_hyperparameters:
                    momentum = float(
                        optimizer_hyperparameters["momentum"].strip())
                else:
                    momentum = 0.0

                if "nesterov" in optimizer_hyperparameters:
                    nesterov = bool(
                        optimizer_hyperparameters["nesterov"].strip())
                else:
                    nesterov = False
                
                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.SGD(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov,
                        centered=centered)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.SGD(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.SGD(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.SGD(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            else:
                raise Exception("unknown optimizer specified: " + optimizer)
        else:
            raise Exception("no optimizer specified")

    @staticmethod
    def get_loss(loss_hyperparameters):
        if "loss" in loss_hyperparameters:
            loss = loss_hyperparameters["loss"].lower().replace(
                "_", "").strip()
            if loss == "binarycrossentropy":
                if "from_logits" in loss_hyperparameters:
                    from_logits = bool(
                        loss_hyperparameters["from_logits"].strip())
                else:
                    from_logits = False

                if "label_smoothing" in loss_hyperparameters:
                    label_smoothing = float(
                        loss_hyperparameters["label_smoothing"].strip())
                else:
                    label_smoothing = 0.0
                return tf.keras.losses.BinaryCrossentropy(
                    from_logits=from_logits,
                    label_smoothing=label_smoothing)
            elif loss == "categoricalcrossentropy":
                if "from_logits" in loss_hyperparameters:
                    from_logits = bool(
                        loss_hyperparameters["from_logits"].strip())
                else:
                    from_logits = False

                if "label_smoothing" in loss_hyperparameters:
                    label_smoothing = float(
                        loss_hyperparameters["label_smoothing"].strip())
                else:
                    label_smoothing = 0.0
                return tf.keras.losses.CategoricalCrossentropy(
                    from_logits=from_logits,
                    label_smoothing=label_smoothing)
            elif loss == "categoricalhinge":
                return tf.keras.losses.CategoricalHinge()
            elif loss == "cosinesimilarity":
                if "axis" in loss_hyperparameters:
                    axis = int(loss_hyperparameters["axis"].strip())
                else:
                    axis = -1
                return tf.keras.losses.CosineSimilarity(axis=axis)
            elif loss == "hinge":
                return tf.keras.losses.Hinge()
            elif loss == "huber":
                if "delta" in loss_hyperparameters:
                    delta = float(
                        loss_hyperparameters["delta"].strip())
                else:
                    delta = 1.0
                return tf.keras.losses.Huber(delta=delta)
            elif loss == "kldivergence" or loss == "kld":
                return tf.keras.losses.KLDivergence()
            elif loss == "logcosh":
                return tf.keras.losses.LogCosh()
            elif loss == "meanabsoluteerror" or loss == "mae":
                return tf.keras.losses.MeanAbsoluteError()
            elif loss == "meanabsolutepercentageerror" or loss == "mape":
                return tf.keras.losses.MeanAbsolutePercentageError()
            elif loss == "meansquarederror" or loss == "mse":
                return tf.keras.losses.MeanSquaredError()
            elif loss == "meansquaredlogarithmicerror" or loss == "msle":
                return tf.keras.losses.MeanSquaredLogarithmicError()
            elif loss == "poisson":
                return tf.keras.losses.Poisson()
            elif loss == "sparsecategoricalcrossentropy":
                if "from_logits" in loss_hyperparameters:
                    from_logits = bool(
                        loss_hyperparameters["from_logits"].strip())
                else:
                    from_logits = False
                return tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=from_logits)
            elif loss == "squaredhinge":
                return tf.keras.losses.SquaredHinge()
            else:
                raise Exception("unknown loss specified: " + loss)
        else:
            raise Exception("no loss specified")
