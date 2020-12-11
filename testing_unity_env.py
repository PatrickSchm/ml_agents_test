from mlagents_envs.environment import UnityEnvironment
import numpy as np
import tensorflow
import time

import tensorflow as tf
import datetime
from tensorflow import keras
from collections import deque
import numpy as npll

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

def epsilon_greedy_policy(obs,model, epsilon=0):
    if np.random.rand() < epsilon:
        ran = np.random.randint(0,4)
        action = np.zeros(4)
        action[ran] = 1

        return np.array([action])
    else:
        Q_values = model.predict(obs)
        return np.array(Q_values)



def sample_experiences(replay_buffer, batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]
    return states, actions, rewards, next_states, dones

def play_one_step(env, model, replay_buffer, obs, epsilon):
    action = epsilon_greedy_policy(obs, model, epsilon)

    act_in = np.argmax(action[0])

    if act_in == 0:
        action_unity = np.array([[1,0]])
    if act_in == 1:
        action_unity = np.array([[-1,0]])
    if act_in == 2:
        action_unity = np.array([[0, 1]])
    if act_in == 3:
        action_unity = np.array([[0, -1]])
    env.set_actions("RollerBall?team=0", action_unity)


    [DecisionSteps, _] = env.get_steps("RollerBall?team=0")
    reward = DecisionSteps.reward
    next_state = DecisionSteps.obs
    env.step()
    [_, TerminalSteps] = env.get_steps("RollerBall?team=0")
    done = int(len(TerminalSteps.interrupted)>0)
    if done:
        reward = TerminalSteps.reward
        obs = TerminalSteps.obs
    replay_buffer.append((obs[0][0], act_in, reward[0], next_state[0][0], done))

    return next_state, done, reward

def training_step(train_loss,train_accuracy,model, target, loss_fn, optimizer, discount_factor, n_output, replay_buffer, batch_size):
    experiences = sample_experiences(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_output).numpy()
    best_next_Q_values = (target.predict(next_states)*next_mask).sum(axis=1)

    target_Q_values = (rewards + (1.0-dones)*discount_factor * best_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_output)

    with tf.GradientTape() as tape:
        Q_values_all = model(states)
        Q_values = tf.reduce_sum(Q_values_all*mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    train_loss(loss)
    train_accuracy(target_Q_values, Q_values)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def train_roller():
    play_mode = False
    load_model = True
    dueling_network = True

    input_shape = [4]
    n_outputs = 4

    state = 0

    batch_size = 20
    discount_factor = 0.99
    optimizer = keras.optimizers.Adam(lr=5.e-4)
    loss_fn = keras.losses.mean_squared_error

    model_out = 'wall_model_double_dueling5000'
    model_name = 'wall_model_5000'
    replay_buffer = deque(maxlen=8000)
    # Define our metrics

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = model_out + 'logs/gradient_tape/' + model_out + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # This is a non-blocking call that only loads the environment.
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=None, seed=1, side_channels=[channel])
    channel.set_configuration_parameters(time_scale=100.0)
    env.step()
    env.set_actions("RollerBall?team=0", np.array([np.random.rand(2)]))


    if load_model or play_mode:
        model = keras.models.load_model(model_name)
        model.compile(optimizer=optimizer,
                      loss=loss_fn)
    else:
        if dueling_network:
            K = keras.backend
            input_states = keras.layers.Input(shape=input_shape)
            hidden1 = keras.layers.Dense(32, activation="elu")(input_states)
            hidden2 = keras.layers.Dense(32, activation="elu")(hidden1)
            state_values = keras.layers.Dense(1)(hidden2)
            raw_advantages = keras.layers.Dense(n_outputs)(hidden2)
            advantages = raw_advantages - K.max(raw_advantages, axis=1, keepdims=True)
            Q_values = state_values + advantages
            model = keras.Model(inputs=[input_states], outputs=[Q_values])
        else:
            model = keras.models.Sequential([
                keras.layers.Dense(64, activation="elu", input_shape=input_shape),
                keras.layers.Dense(64, activation="elu"),

                keras.layers.Dense(n_outputs)
            ])
    target = keras.models.clone_model(model)
    for episode in range(1000000):
        env.reset()
        env.step()

        if episode %100 <= 2:
            channel.set_configuration_parameters(time_scale=1.0)
        else:
            channel.set_configuration_parameters(time_scale=1000)


        [DecisionSteps, _] = env.get_steps("RollerBall?team=0")
        obs = DecisionSteps.obs
        rewards = []
        steps=0
        while True:
            steps+=1
            if play_mode:
                epsilon = 0
                channel.set_configuration_parameters(time_scale=1.0)
            elif load_model:
                epsilon = max(0.75 - (episode-50) / 10000.*0.75, 0.1)
            elif episode>100:
                epsilon = max(1 - (episode-100) / 2500., 0.05)
            else:
                epsilon = 1

            obs, done, reward = play_one_step(env, model, replay_buffer, obs, epsilon)
            rewards.append(reward)
            if done:
                break
        if episode % 1 == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=episode)
                tf.summary.scalar('reward_sum', np.sum(np.array(rewards)), step=episode)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=episode)
        if episode >50 and play_mode == False:
            training_step(train_loss,train_accuracy ,model, target, loss_fn, optimizer, discount_factor, n_outputs, replay_buffer, batch_size)
            template = 'Epoch {}, Steps {}, Epsilon {}, Loss: {}, Accuracy: {}, RewardSum: {}'
            print(template.format(episode + 1,
                                  steps,
                                  epsilon,
                                  train_loss.result(),
                                  train_accuracy.result() * 100,
                                  np.sum(np.array(rewards))))
            train_loss.reset_states()
            train_accuracy.reset_states()
        else:
            template = 'Epoch {}, Steps{}, Epsilon {}, Loss: {}, Accuracy: {}, RewardSum: {}'
            print(template.format(episode + 1,
                                  steps,
                                  epsilon,
                                  0,
                                  0,
                                  np.sum(np.array(rewards))))


        if episode %500 == 0 and play_mode == False:
            target.set_weights(model.get_weights())
            keras.models.save_model(model, model_out+str(episode))
        #

