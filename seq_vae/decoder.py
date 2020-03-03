import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.layers import core as layers_core

import help.lstm_utils
from help.utils import flatten_maybe_padded_sequences

seq2seq = tf.contrib.seq2seq


class LstmDecoder(object):

    def __init__(self, hparams, name_or_scope='decoder'):
        self.params = None
        self.name_or_scope = name_or_scope
        self._max_seq_len = hparams.max_seq_len
        self._batch_size = hparams.batch_size
        self._output_depth = hparams.feature_dim
        self._z_size = hparams.z_size
        self._update_rate = hparams.dec_update_rate

        self._output_layer = layers_core.Dense(self._output_depth, name='output_projection')

        self._dec_cell = help.lstm_utils.rnn_cell(
            hparams.dec_rnn_size,
            hparams.dropout_keep_prob,
            hparams.residual_decoder)

    def reconstruct_loss(self, z, target_seq):
        seq = tf.pad(target_seq[:, :self._max_seq_len - 1], [(0, 0), (1, 0), (0, 0)])
        seq_len = [self._max_seq_len] * self._batch_size
        train_helper = seq2seq.TrainingHelper(seq, seq_len)

        with tf.variable_scope(self.name_or_scope, reuse=tf.AUTO_REUSE):
            init_state = help.lstm_utils.initial_cell_state_from_embedding(
                self._dec_cell, z,
                name='z_to_state')
            train_decoder = help.lstm_utils.Seq2SeqLstmDecoder(
                self._dec_cell,
                train_helper,
                initial_state=init_state,
                input_shape=train_helper.inputs.shape[2:],
                output_layer=self._output_layer)
            train_output, _, train_lengths = seq2seq.dynamic_decode(
                train_decoder,
                maximum_iterations=self._max_seq_len)
            flat_train_output = flatten_maybe_padded_sequences(
                train_output.rnn_output,
                train_lengths)
            flat_target = flatten_maybe_padded_sequences(target_seq, seq_len)
            # reconstruction loss
            r_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=flat_target,
                logits=flat_train_output)
            # sum loss over sequences.
            cum_x_len = tf.concat([(0,), tf.cumsum(seq_len)], axis=0)
            r_losses = []
            for i in range(self._batch_size):
                b, e = cum_x_len[i], cum_x_len[i + 1]
                r_losses.append(tf.reduce_sum(r_loss[b:e]))
            r_loss = tf.stack(r_losses)
            self.r_loss = tf.reduce_mean(r_loss)

            # summary r_loss
            tf.summary.scalar("r_loss", self.r_loss)
            # accuracy
            flat_truth = tf.argmax(flat_target, axis=1)
            flat_predictions = tf.argmax(flat_train_output, axis=1)
            self.train_accuracy_real = tf.reduce_sum(
                tf.cast(tf.equal(flat_truth, flat_predictions), tf.float32)) / tf.cast(tf.shape(flat_target)[0],
                                                                                       tf.float32)
            tf.summary.scalar("train_accuracy_real", self.train_accuracy_real)

        if self.params is None:
            self.params = [param for param in tf.trainable_variables() if param.name.startswith(self.name_or_scope)]
        return self.r_loss

    def generate_part(self, z, sequence, part_num):
        part_num_tensor = tf.constant(part_num, name='part_num')
        size = z.shape[0]
        start_inputs = tf.zeros([size, self._output_depth], dtype=tf.float32)
        if sequence is not None:
            sequence_transpose = tf.transpose(sequence, [1, 0, 2])
        else:
            sequence_transpose = tf.zeros(shape=(self._max_seq_len, size, self._output_depth))

        def init_fn():
            return tf.zeros(size, tf.bool), start_inputs

        def sample_part_fn(time, outputs, state):
            sample_ids = tf.cond(
                tf.less(time, part_num_tensor),
                lambda: sequence_transpose[time],
                lambda: tfp.distributions.OneHotCategorical(logits=outputs, dtype=tf.float32).sample())
            return sample_ids

        def next_inputs_fn(time, outputs, state, sample_ids):
            return False, sample_ids, state

        gen_part_helper = seq2seq.CustomHelper(
            initialize_fn=init_fn,
            sample_fn=sample_part_fn,
            next_inputs_fn=next_inputs_fn,
            sample_ids_shape=[self._output_depth],
            sample_ids_dtype=tf.float32)

        with tf.variable_scope(self.name_or_scope, reuse=tf.AUTO_REUSE):
            init_state = help.lstm_utils.initial_cell_state_from_embedding(
                self._dec_cell, z,
                name='z_to_state')
            gen_part_decoder = help.lstm_utils.Seq2SeqLstmDecoder(
                self._dec_cell,
                gen_part_helper,
                initial_state=init_state,
                input_shape=self._output_depth,
                output_layer=self._output_layer)
            gen_part_output, _, _ = seq2seq.dynamic_decode(
                gen_part_decoder,
                maximum_iterations=self._max_seq_len)
            self.gen_part_output = gen_part_output.sample_id
            self.gen_part_output_sm = tf.nn.softmax(gen_part_output.rnn_output)

        if self.params is None:
            self.params = [param for param in tf.trainable_variables() if param.name.startswith(self.name_or_scope)]
        return self.gen_part_output, self.gen_part_output_sm

    def generate(self, z):
        return self.generate_part(z, None, 0)

    def collect_all_variables(self):
        vars = self.params
        vars_dict = {}
        for v in vars:
            name = v.name[v.name.find(self.name_or_scope) + len(self.name_or_scope):]
            vars_dict[name] = v
        return vars_dict

    def update(self, vars_dict):
        old_variables_dict = self.collect_all_variables()
        update_ops = []

        for name, v in old_variables_dict.items():
            update_ops.append(tf.assign(v, value=vars_dict[name] * self._update_rate + v * (1 - self._update_rate)))
        return update_ops
