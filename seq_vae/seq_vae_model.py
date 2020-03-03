import tensorflow as tf
import tensorflow_probability as tfp

ds = tfp.distributions


class SeqVAE(object):

    def __init__(self, hparams, encoder, decoder_theta, decoder_beta, discriminator, global_step):
        self._max_seq_len = hparams.max_seq_len
        self._free_bits = hparams.free_bits
        self._rollout_num = hparams.rollout_num
        self._step = hparams.compute_rewards_step
        self._max_beta = hparams.max_beta
        self._beta_decay_rate = hparams.beta_decay_rate
        self._dropout_keep_prob = hparams.dropout_keep_prob

        self.encoder = encoder
        self.decoder_theta = decoder_theta
        self.decoder_beta = decoder_beta
        self.discriminator = discriminator

        self.z_p = ds.MultivariateNormalDiag(
            loc=[0.] * hparams.z_size,
            scale_diag=[1.] * hparams.z_size)

        self._global_step = global_step

    def _compute_kl_loss(self, target_seq, seq_len):
        z_q = self.encoder.encode(target_seq, seq_len)
        # compute formula:https://zhuanlan.zhihu.com/p/22464760
        self._kl_div = ds.kl_divergence(z_q, self.z_p)
        # <Improving variational inference with inverse auto regressive flow>, NIPS 2016
        # Bits to exclude from KL loss per dimension.
        free_nats = self._free_bits * tf.math.log(2.0)
        self.kl_loss = tf.reduce_mean(tf.maximum(self._kl_div - free_nats, 0))

        tf.summary.scalar("seq_vae/kl_bits", tf.reduce_mean(self._kl_div / tf.math.log(2.0)))
        tf.summary.scalar("seq_vae/kl_loss", self.kl_loss)

        return z_q

    def _get_rewards_for_pg_loss(self, z, target_seq):
        rewards = []
        for k in range(self._rollout_num):
            count = 0
            for i in range(self._step, self._max_seq_len, self._step):
                gen_part_output, _ = self.decoder_beta.generate_part(z, target_seq, i)
                scores_sm, _ = self.discriminator.discriminate(gen_part_output)
                reward = tf.squeeze(scores_sm[:, :1], 1)
                if k == 0:
                    rewards.append(reward)
                else:
                    rewards[count] += reward
                count += 1

            scores_sm, _ = self.discriminator.discriminate(target_seq)
            reward = tf.squeeze(scores_sm[:, :1], 1)
            if k == 0:
                rewards.append(reward)
            else:
                rewards[count] += reward

        rewards = [reward / (1.0 * self._rollout_num) for reward in rewards]
        rewards = tf.transpose(tf.convert_to_tensor(rewards), perm=[1, 0])  # batch_size * (max_seq_len // step)
        return rewards

    def _compute_pg_loss(self, z, target_seq):
        _, output_sm = self.decoder_theta.generate_part(z, target_seq, self._max_seq_len)

        rewards = self._get_rewards_for_pg_loss(z, target_seq)

        # make log(output_sm) not get nan
        output_sm = tf.clip_by_value(output_sm, 1e-20, 1.0)
        log_signal = tf.reduce_sum(target_seq * tf.log(output_sm), axis=2)  # batch_size * max_seq_len
        log_sum = []
        step = self._step
        for i in range(self._max_seq_len // step):
            log_sum.append(tf.reduce_sum(log_signal[:, :(i + 1) * step], axis=1))
        log_sum = tf.transpose(tf.convert_to_tensor(log_sum), perm=[1, 0])

        pg_loss = tf.reduce_sum(log_sum * rewards, axis=1)  # batch_size
        self.pg_loss = - tf.reduce_mean(pg_loss)

        tf.summary.scalar("seq_vae/pg_loss", self.pg_loss)

    def compute_loss(self, target_seq, seq_len):
        # kl_loss
        z_q = self._compute_kl_loss(target_seq, seq_len)
        z = z_q.sample()

        # r_loss
        self.r_loss = self.decoder_theta.reconstruct_loss(z, target_seq)
        tf.summary.scalar("seq_vae/r_loss", self.r_loss)

        # PG loss
        self._compute_pg_loss(z, target_seq)

        # VAE loss
        kl_loss_beta = (1.0 - tf.pow(self._beta_decay_rate, tf.to_float(self._global_step))) * self._max_beta
        self.vae_loss = kl_loss_beta * self.kl_loss + self.r_loss

        return self.vae_loss, self.pg_loss

    def discriminator_loss(self, inputs, labels):
        _, self.dis_loss = self.discriminator.discriminate(inputs, labels)
        return self.dis_loss

    def generate(self, z):
        return self.decoder_theta.generate(z)

    def update_decoder_beta(self):
        return self.decoder_beta.update(self.decoder_theta.collect_all_variables())
