import tensorflow as tf


class LastEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir: str = output_dir

    def on_epoch_end(self, epoch, logs=None):
        with open(self.output_dir + "last-epoch-completed.txt", "w") as last_epoch_file:
            last_epoch_file.write(str(epoch + 1) + "\n")
