import tensorflow as tf
from .networks import Actor, Critic
from .buffer import Buffer, PriorityBuffer
import os

#Buffer, PriorityBuffer = buffer.Buffer, buffer.PriorityBuffer
#Actor, Critic = networks.Actor, networks.Critic


class DDPG(tf.keras.models.Model):
    def __init__(self,
                 states=3,
                 actions=1,
                 units=(512, 512),
                 activation="relu",
                 optimizer="Adam",
                 buffer_size=1e6,
                 batch_size=32,
                 tau=0.001,
                 gamma=0.99
                 ):
        super().__init__()
        self.buffer_size = buffer_size
        self.actions = actions
        self.states = states
        self.units = units
        self.activation = activation
        self.tau = tau
        self.gamma = gamma
        self.buffer = Buffer(states, actions, 1, max_size=buffer_size, batch_size=batch_size)
        self.actor = Actor(states, actions, units=units, activation=activation, optimizer=optimizer, name="Actor")
        self.actor_target = Actor(states, actions, units=units, activation=activation, optimizer=optimizer,
                                  name="ActorTarget")
        self.critic = Critic(states, actions, units=units, activation=activation, optimizer=optimizer, name="Critic")
        self.critic_target = Critic(states, actions, units=units, activation=activation, optimizer=optimizer,
                                    name="CriticTarget")

    def call(self, inputs, training=None, mask=None, target=False, exploration=False):
        s = inputs
        if target:
            a = self.actor_target(s, exploration=exploration)
            q = self.critic_target([s, a])
        else:
            a = self.actor(s, exploration=exploration)
            q = self.critic([s, a])
        return a, q

    def train_step(self, data=None):
        s0, a, r, s1, d = self.buffer.sample()
        s0 = tf.convert_to_tensor(s0)
        a = tf.convert_to_tensor(a)
        r = tf.convert_to_tensor(r)
        s1 = tf.convert_to_tensor(s1)
        d = tf.convert_to_tensor(d)

        with tf.GradientTape() as tape:
            critic_loss = tf.reduce_mean(self.compute_td(s0, a, r, s1, d) ** 2)
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            a = self.actor(s0, exploration=False)
            q = self.critic([s0, a])
            actor_loss = self.actor_loss(q)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        self.update_targets()

    def compute_td(self, s0, a, r, s1, d):
        q = self.critic([s0, a])
        a_target = self.actor_target(s1)
        q_target = self.critic_target([s1, a_target])
        return self.td_error(q, q_target, r, d)

    def td_error(self, q_current, q_target, r, done):
        target = r + self.gamma * q_target * (1 - done)
        td = target - q_current
        return td

    def critic_loss(self, q_current, q_target, r, done):
        td = self.td_error(q_current, q_target, r, done)
        return tf.reduce_mean(td ** 2)

    @staticmethod
    def actor_loss(q_values):
        return -tf.reduce_mean(q_values)

    def store_transition(self, *args):
        self.buffer.store_transition(*args)

    def update_targets(self, tau=None):
        if tau is not None:
            w = tau
        else:
            w = self.tau
        # Update target critic
        for target_weights, current_weights in zip(self.critic_target.trainable_variables,
                                                   self.critic.trainable_variables):
            target_weights.assign(current_weights * w + target_weights * (1 - w))
        # Update target actor
        for target_weights, current_weights in zip(self.actor_target.trainable_variables,
                                                   self.actor.trainable_variables):
            target_weights.assign(current_weights * w + target_weights * (1 - w))

    def save_models(self, dir):
        self.actor.save(os.path.join(dir, "actor"))
        self.actor_target.save(os.path.join(dir, "actor_target"))
        self.critic.save(os.path.join(dir, "critic"))
        self.critic_target.save(os.path.join(dir, "critic_target"))

    def load_models(self, dir):
        self.actor = tf.keras.models.load_model(os.path.join(dir, "actor"))
        self.actor_target = tf.keras.models.load_model(os.path.join(dir, "actor_target"))
        self.critic = tf.keras.models.load_model(os.path.join(dir, "critic"))
        self.critic_target = tf.keras.models.load_model(os.path.join(dir, "critic_target"))

    def save_buffer(self, dir):
        self.buffer.save(dir)

    def load_buffer(self, dir):
        self.buffer.load(dir)


class DDPGPriority(DDPG):
    def __init__(self, alpha, beta, states=3,
                 actions=1,
                 units=(512, 512),
                 activation="relu",
                 optimizer="Adam",
                 buffer_size=1e6,
                 batch_size=32,
                 tau=0.001,
                 gamma=0.99,
                 **kwargs):
        super().__init__(units=units, activation=activation, optimizer=optimizer, tau=tau, gamma=gamma, **kwargs)
        self.buffer = PriorityBuffer(alpha, beta, state_shape=states, action_shape=actions, reward_shape=1,
                                     max_size=buffer_size, batch_size=batch_size)

    def train_step(self, data=None):
        s0, a, r, s1, d, w = self.buffer.sample()
        s0 = tf.convert_to_tensor(s0)
        a = tf.convert_to_tensor(a)
        r = tf.convert_to_tensor(r)
        s1 = tf.convert_to_tensor(s1)
        d = tf.convert_to_tensor(d)
        w = tf.convert_to_tensor(w)
        with tf.GradientTape() as tape:
            q = self.critic([s0, a])
            a_target = self.actor_target(s1)
            q_target = self.critic_target([s1, a_target])
            critic_loss = self.critic_loss(q, q_target, r, d, w)
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            a = self.actor(s0, exploration=False)
            q = self.critic([s0, a])
            actor_loss = self.actor_loss(q)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        self.update_targets()

    def critic_loss(self, q_current, q_target, r, done, w):
        target = r + self.gamma * q_target * (1 - done)
        return tf.reduce_mean(w * (target - q_current) ** 2)

    def store_transition(self, *args):
        args = [tf.cast(tf.expand_dims(arg, 0), tf.float32) for arg in args]
        td = tf.squeeze(tf.abs(self.compute_td(*args)))
        self.buffer.store_transition(*args, td.numpy())
