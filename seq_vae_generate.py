import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from magenta import music as mm

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
flags.DEFINE_string(
    'output_dir', 'create/',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_integer(
    'num', 1,
    'The number of music which you will want to generate.')
flags.DEFINE_string(
    'mode', 'sample',
    'Generate mode (either `sample` or `test` or `interpolate`).')
flags.DEFINE_string(
    'input_midi_1', None,
    'Path of start MIDI file for interpolation.')
flags.DEFINE_string(
    'input_midi_2', None,
    'Path of end MIDI file for interpolation.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')

# GPU's configuration
# gpu_config = tf.ConfigProto()
# gpu_config.gpu_options.allow_growth = True

# only use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def generate():
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

    # params
    z_size = hparams.z_size
    batch_size = hparams.batch_size

    graph = tf.get_default_graph()
    with graph.as_default():
        sess = tf.Session()

        encoder = BidirectionalLstmEncoder(hparams, name_or_scope='seq_vae/encoder')
        decoder_theta = LstmDecoder(hparams, name_or_scope='seq_vae/decoder')
        decoder_beta = LstmDecoder(hparams, name_or_scope='seq_vae-copy/decoder')
        dis = BidirectionalLstmDiscriminator(hparams, name_or_scope='seq_vae/rnn_discriminator')
        seq_vae = SeqVAE(hparams, encoder, decoder_theta, decoder_beta, dis, None)

        if FLAGS.mode == 'interpolate':
            if FLAGS.input_midi_1 is None or FLAGS.input_midi_2 is None:
                raise ValueError(
                    '`--input_midi_1` and `--input_midi_2` must be specified in '
                    '`interpolate` mode.')
            logging.info('Interpolating...')
            input_midi_1 = os.path.expanduser(FLAGS.input_midi_1)
            input_midi_2 = os.path.expanduser(FLAGS.input_midi_2)
            input_1 = mm.midi_file_to_note_sequence(input_midi_1)
            input_2 = mm.midi_file_to_note_sequence(input_midi_2)

            inputs = []
            lengths = []
            for note_sequence in [input_1, input_2]:
                extracted_tensors = configs.mel_16bar_converter.to_tensors(note_sequence)
                if len(extracted_tensors.inputs) > 0:
                    inputs.append(extracted_tensors.inputs[0])
                    lengths.append(extracted_tensors.lengths[0])
            inputs = np.array(inputs).astype(np.float)
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
            mu = encoder.get_mu(inputs, lengths)

        # generate_op
        # used to sample z
        z = tfd.MultivariateNormalDiag(
            loc=[0] * z_size,
            scale_diag=[1] * z_size).sample(batch_size)
        generate_op, _ = seq_vae.generate(z)

        saver = tf.train.Saver()
        sess.run(tf.local_variables_initializer())

        # load trained
        save_path = tf.train.latest_checkpoint(checkpoints_dir)
        logging.info('Load model from %s...' % save_path)
        saver.restore(sess, save_path)

        checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
        meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
        step = int(meta_graph_path.split("-")[-1].split(".")[0])

        logging.info('Start...')

        num = FLAGS.num
        count = 0
        # generate music
        if FLAGS.mode == 'sample':
            basename = os.path.join(FLAGS.run_dir + FLAGS.output_dir, '%s-*.mid' % FLAGS.config)
            while count < num:
                f_ms = config.data_converter.to_items(sess.run(generate_op))
                for f_m in f_ms:
                    mm.sequence_proto_to_midi_file(f_m, basename.replace('*', '%03d' % count))
                    count += 1
                    if count == num:
                        break
        elif FLAGS.mode == 'test':
            save_file_name = os.path.join(FLAGS.run_dir + FLAGS.output_dir,
                                          FLAGS.config + '_' + str(step // 1000) + 'k.npy')
            results = []
            while count < num:
                f_ms = sess.run(generate_op)
                for f_m in f_ms:
                    results.append(f_m)
                    count += 1
                    if count == num:
                        break
            np.save(save_file_name, results)
        elif FLAGS.mode == 'interpolate':
            mu_values = sess.run(mu)
            z = np.array([utils.slerp(mu_values[0], mu_values[1], t) for t in np.linspace(0, 1, num)])
            results = sess.run(seq_vae.generate(z)[0])
            note_sequence_arr = config.data_converter.to_items(results)
            basename = os.path.join(FLAGS.run_dir + FLAGS.output_dir, 'interpolate-*.mid')
            while count < num:
                mm.sequence_proto_to_midi_file(note_sequence_arr[count], basename.replace('*', '%03d' % count))
                count += 1

        logging.info('Done.')


def main(unused_argv):
    tf.logging.set_verbosity(FLAGS.log)
    generate()


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
