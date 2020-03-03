import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import configs
from help import utils
from seq_vae.decoder import LstmDecoder
from seq_vae.discriminator_rnn import BidirectionalLstmDiscriminator
from seq_vae.encoder import BidirectionalLstmEncoder
from seq_vae.seq_vae_model import SeqVAE

tfd = tfp.distributions
logging = tf.logging
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'run_dir', 'output/',
    'Path where checkpoints and summary events will be located during '
    'training and evaluation.')
flags.DEFINE_string(
    'config', 'SeqVAE',
    'The name of the config to use.')
flags.DEFINE_string(
    'load_model', None,
    'folder of saved model that you wish to continue training, default: None')
flags.DEFINE_integer(
    'exp_num', 5,
    'Specify the number of experiments.')
flags.DEFINE_integer(
    'num', 10,
    'The number of music which you will want to interpolation.')
flags.DEFINE_string(
    'input_midis', None,
    'MIDIs(*.npy) file for interpolation.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')

# GPU's configuration
# gpu_config = tf.ConfigProto()
# gpu_config.gpu_options.allow_growth = True


# only use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def compute_hamming_distance(arr):
    standard = np.argmax(arr[0], 1)
    result = []
    for i in range(2, len(arr)):
        num = np.sum(np.equal(standard, np.argmax(arr[i], 1)))
        result.append(num)
    return result


def interpolate():
    if FLAGS.run_dir is None:
        raise ValueError('You must specify `run_dir`!')
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train/')
    if FLAGS.load_model is None:
        raise ValueError('You must specify `load_model`!')
    checkpoints_dir = train_dir + FLAGS.load_model

    # configuration
    config = configs.CONFIG_MAP[FLAGS.config]
    hparams = config.hparams
    hparams.dropout_keep_prob = 1.0

    batch_size = hparams.batch_size
    max_seq_len = hparams.max_seq_len
    z_size = hparams.z_size

    graph = tf.get_default_graph()
    with graph.as_default():
        sess = tf.Session()

        encoder = BidirectionalLstmEncoder(hparams, name_or_scope='seq_vae/encoder')
        decoder_theta = LstmDecoder(hparams, name_or_scope='seq_vae/decoder')
        decoder_beta = LstmDecoder(hparams, name_or_scope='seq_vae-copy/decoder')
        dis = BidirectionalLstmDiscriminator(hparams, name_or_scope='seq_vae/rnn_discriminator')
        seq_vae = SeqVAE(hparams, encoder, decoder_theta, decoder_beta, dis, None)

        if FLAGS.input_midis is None:
            raise ValueError('Please specify `input_midis`!')
        if FLAGS.num > batch_size:
            raise ValueError('Maximum number of interpolation supported is the number of `batch_size`')
        logging.info('Start...')

        midis = np.load(FLAGS.input_midis)

        inputs = tf.placeholder(dtype=tf.float32, shape=(2, max_seq_len, hparams.feature_dim))

        mu = encoder.get_mu(inputs, [max_seq_len] * 2)

        z = tf.placeholder(dtype=tf.float32, shape=(FLAGS.num, hparams.z_size))
        generate_op, _ = seq_vae.generate(z)

        saver = tf.train.Saver()
        sess.run(tf.local_variables_initializer())

        # load trained
        save_path = tf.train.latest_checkpoint(checkpoints_dir)
        logging.info('Load model from %s...' % save_path)
        saver.restore(sess, save_path)

        logging.info('Start...')

        num = FLAGS.num
        exp_num = FLAGS.exp_num
        exp_count = 0
        exp_record = [0] * num

        while exp_count < exp_num:
            index_1 = random.randint(0, len(midis) - 1)
            index_2 = random.randint(0, len(midis) - 1)
            while index_2 == index_1:
                index_2 = random.randint(0, len(midis) - 1)
            interpolate_curr_arr = [midis[index_1], midis[index_2]]
            inputs_v = np.array([midis[index_1], midis[index_2]])

            mu_values = sess.run(mu, feed_dict={inputs: inputs_v})
            z_v = np.array([utils.slerp(mu_values[0], mu_values[1], t) for t in np.linspace(0, 1, num)])
            generate_results = sess.run(generate_op, feed_dict={z: z_v})
            interpolate_curr_arr.extend(generate_results[:num])

            curr_record = compute_hamming_distance(interpolate_curr_arr)
            for i in range(num):
                exp_record[i] += curr_record[i]

            print(exp_count)
            exp_count += 1

        print("===== RESULT =====")
        print(exp_record)
        exp_record = [(max_seq_len * exp_num - item) / (max_seq_len * exp_num) for item in exp_record]
        print(exp_record)
        print("==================")

        logging.info('Done!!!')


def main(unused_argv):
    tf.logging.set_verbosity(FLAGS.log)
    interpolate()


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
