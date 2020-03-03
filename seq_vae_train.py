import os
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp

import configs
from help.utils import data_confusion
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
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')

# GPU's configuration
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

# only use CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# ==============================================================

from help import data


def dataset_fn(config, num_threads=4, is_training=True, cache_dataset=True):
    """对 Magenta 项目脚本（通过编码MIDI文件）生成的 tfrecord 文件读取的预处理
    Note：本函数是对 data 的封装，需要 config 具备一系列属性以满足 data 中的要求
    """
    return data.get_dataset(
        config,
        tf_file_reader=tf.data.TFRecordDataset,
        num_threads=num_threads,
        is_training=is_training,
        cache_dataset=cache_dataset)


def get_input_tensors_from_dataset(dataset, batch_size, max_seq_len, pitch_num, control_num=0):
    """从 Magenta 项目脚本（通过编码MIDI文件）生成的 tfrecord 文件读取数据

    :param dataset: 一般为上述 dataset_fn() 的输出
    :param batch_size: batch size
    :param max_seq_len: 序列的最大长度
    :param pitch_num: 音高的维数
    :param control_num: 一般为和弦的维数
    :return: （input_sequence, output_sequence, control_sequence, sequence_length）
    """
    iterator = dataset.make_one_shot_iterator()
    input_sequence, output_sequence, control_sequence, sequence_length = iterator.get_next()

    input_sequence.set_shape([batch_size, max_seq_len, pitch_num])
    input_sequence = tf.to_float(input_sequence)

    output_sequence.set_shape([batch_size, max_seq_len, pitch_num])
    output_sequence = tf.to_float(output_sequence)

    sequence_length.set_shape([batch_size] + sequence_length.shape[1:].as_list())
    sequence_length = tf.minimum(sequence_length, max_seq_len)

    if control_num != 0:
        control_sequence.set_shape([batch_size, max_seq_len, control_num])
        control_sequence = tf.to_float(control_sequence)
    else:
        control_sequence = None
    return input_sequence, output_sequence, control_sequence, sequence_length


def get_data(config):
    hparams = config.hparams
    batch_size = hparams.batch_size
    max_seq_len = hparams.max_seq_len
    feature_dim = hparams.feature_dim
    true_data, _, _, seq_len = get_input_tensors_from_dataset(
        dataset_fn(config),
        batch_size, max_seq_len, feature_dim)
    target_seq = true_data[:, :max_seq_len]
    return target_seq, seq_len


# ==============================================================

def get_checkpoints_dir():
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train/')
    checkpoints_dir = train_dir + FLAGS.config + "-{}".format(current_time)
    if FLAGS.load_model is not None:
        checkpoints_dir = train_dir + FLAGS.load_model
    return checkpoints_dir


def get_update_op(loss, var_list, optimizer, grad_clip, global_step=None):
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, var_list), grad_clip)
    return optimizer.apply_gradients(zip(grads, var_list), global_step=global_step)


def train():
    checkpoints_dir = get_checkpoints_dir()

    # configuration
    config = configs.CONFIG_MAP[FLAGS.config]
    hparams = config.hparams

    batch_size = hparams.batch_size
    z_size = hparams.z_size

    graph = tf.get_default_graph()
    with graph.as_default():
        sess = tf.Session()
        global_step = tf.train.get_or_create_global_step()

        encoder = BidirectionalLstmEncoder(hparams, name_or_scope='seq_vae/encoder')
        decoder_theta = LstmDecoder(hparams, name_or_scope='seq_vae/decoder')
        decoder_beta = LstmDecoder(hparams, name_or_scope='seq_vae-copy/decoder')
        dis = BidirectionalLstmDiscriminator(hparams, name_or_scope='seq_vae/rnn_discriminator')
        seq_vae = SeqVAE(hparams, encoder, decoder_theta, decoder_beta, dis, global_step)

        target_seq, seq_len = get_data(config)
        z = tfd.MultivariateNormalDiag(
            loc=[0] * z_size,
            scale_diag=[1] * z_size).sample(batch_size)

        generate_op, _ = seq_vae.generate(z)

        inputs, labels = data_confusion(batch_size, generate_op, target_seq)

        # loss
        vae_loss, pg_loss = seq_vae.compute_loss(target_seq, seq_len)
        dis_loss = seq_vae.discriminator_loss(inputs, labels)

        # optimizer
        lr = ((hparams.learning_rate - hparams.min_learning_rate) *
              tf.pow(hparams.decay_rate, tf.to_float(global_step)) +
              hparams.min_learning_rate)
        vae_optimizer = tf.train.AdamOptimizer(lr)
        pg_optimizer = tf.train.AdamOptimizer(lr)
        if hparams.dis_learning_rate == 0.0:
            hparams.dis_learning_rate = lr
        dis_optimizer = tf.train.AdamOptimizer(hparams.dis_learning_rate)

        # use vae_loss to update "encoder" & "decoder"
        vae_loss_update_op = get_update_op(vae_loss, encoder.params + decoder_theta.params,
                                           vae_optimizer, hparams.grad_clip)
        pg_loss_update_op = get_update_op(pg_loss, decoder_theta.params,
                                          pg_optimizer, hparams.grad_clip, global_step=global_step)

        # update "discriminator"
        dis_update_op = get_update_op(dis_loss, dis.params, dis_optimizer, hparams.grad_clip)

        # update decoder beta
        update_op_decoder_beta = seq_vae.update_decoder_beta()

        # merge summary & save
        merge_summary = tf.summary.merge_all(scope='seq_vae')
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver(max_to_keep=0)

        if FLAGS.load_model is None:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            step = 0
        else:
            # load trained
            save_path = tf.train.latest_checkpoint(checkpoints_dir)
            logging.info('Load model from %s...' % save_path)
            saver.restore(sess, save_path)
            # set step
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            step = int(meta_graph_path.split("-")[-1].split(".")[0])
            step += 1

        # make the calculation graph immutable
        graph.finalize()

        while True:
            if step % hparams.dis_train_freq == 0:
                sess.run(dis_update_op)

            kl_loss, r_loss, pg_loss, _, _ = sess.run(
                [seq_vae.kl_loss, seq_vae.r_loss, seq_vae.pg_loss, vae_loss_update_op, pg_loss_update_op])

            # update decoder beta
            sess.run(update_op_decoder_beta)

            # log
            if step % 100 == 0:
                kl_loss, r_loss, pg_loss, summary_vae_pg = sess.run(
                    [seq_vae.kl_loss, seq_vae.r_loss, seq_vae.pg_loss, merge_summary])
                train_writer.add_summary(summary_vae_pg, step)
                train_writer.flush()

                logging.info('   kl_loss  : {}'.format(kl_loss))
                logging.info('   r_loss  : {}'.format(r_loss))
                logging.info('   pg_loss   : {}'.format(pg_loss))

            # save model
            if step % 1000 == 0:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)

            step += 1


def main(unused_argv):
    train()


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
