import tensorflow as tf

import help.lstm_utils as lstm_utils

rnn = tf.contrib.rnn


class BidirectionalLstmDiscriminator(object):

    def __init__(self, hparams, name_or_scope='rnn_discriminator'):
        self.params = None
        self.name_or_scope = name_or_scope
        self._batch_size = hparams.batch_size
        self._max_seq_len = hparams.max_seq_len

        cells_fw = []
        cells_bw = []
        for i, layer_size in enumerate(hparams.dis_rnn_size):
            cells_fw.append(
                lstm_utils.rnn_cell(
                    [layer_size],
                    hparams.dropout_keep_prob,
                    hparams.residual_encoder))
            cells_bw.append(
                lstm_utils.rnn_cell(
                    [layer_size],
                    hparams.dropout_keep_prob,
                    hparams.residual_encoder))

        self._cells_fw = cells_fw
        self._cells_bw = cells_bw

    def discriminate(self, inputs, labels=None):
        sequence_length = [self._max_seq_len] * self._batch_size
        with tf.variable_scope(self.name_or_scope, reuse=tf.AUTO_REUSE):
            _, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(
                self._cells_fw,
                self._cells_bw,
                inputs,
                sequence_length=sequence_length,
                time_major=False,
                dtype=tf.float32, )

            last_c_fw = states_fw[-1][-1].c
            last_c_bw = states_bw[-1][-1].c
            output = tf.concat([last_c_fw, last_c_bw], 1)

            scores = tf.layers.dense(
                output,
                2,
                name='scores',
                kernel_initializer=tf.random_normal_initializer(stddev=0.001))
            self.scores_sm = tf.nn.softmax(scores)
            predictions = tf.argmax(scores, 1, name="predictions")

            if labels is not None:
                # CalculateMean cross-entropy loss
                with tf.variable_scope("loss"):
                    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=labels)
                    self.loss = tf.reduce_mean(losses)
                    tf.summary.scalar("d_loss", self.loss)
                # accuracy
                self.accuracy = tf.reduce_sum(labels * tf.one_hot(predictions, 2)) / self._batch_size
                tf.summary.scalar("d_accuracy", self.accuracy)
                if self.params is None:
                    self.params = [param for param in tf.trainable_variables() if
                                   param.name.startswith(self.name_or_scope)]

        if labels is None:
            return self.scores_sm, None
        else:
            return self.scores_sm, self.loss
