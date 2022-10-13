import gym
import numpy as np

from collections import deque
from time import perf_counter, strftime, localtime
import os

from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam

from preprocess import AtariPreprocessor

if not os.getcwd().endswith(r'src'):
    os.chdir(r'src')
base = os.path.join(os.getcwd(), r'../')


class DQNAgent:
    def __init__(
        self, env, replay_buffer_size=100000,
        batch_size=32, checkpoint=r'saves/pong.tf',
        reward_buffer_size=100, epsilon_start=1,
        epsilon_end=0.01, frame_skips=4,
        resize_shape=(84, 84), state_buffer_size=2,
    ):
        """
        Initialize agent settings.
        Args:
            env: gym environment that returns states as atari frames.
            replay_buffer_size: Size of the replay buffer that will hold record the
                last n observations in the form of (state, action, reward, done, new state)
            batch_size: Training batch size.
            checkpoint: Path to .tf filename under which the trained model will be saved.
            reward_buffer_size: Size of the reward buffer that will hold the last n total
                rewards which will be used for calculating the mean reward.
            epsilon_start: Start value of epsilon that regulates exploration during training.
            epsilon_end: End value of epsilon which represents the minimum value of epsilon
                which will not be decayed further when reached.
            frame_skips: Number of frame skips to use per environment step.
            resize_shape: (m, n) dimensions for the frame preprocessor
            state_buffer_size: Size of the state buffer used by the frame preprocessor.
        """

        self.env = AtariPreprocessor(gym.make(env, full_action_space=False))
        self.input_shape = self.env.observation_space.shape
        self.main_model = self.create_model()
        self.target_model = self.create_model()
        self.buffer_size = replay_buffer_size
        self.buffer = []
        self.batch_size = batch_size
        self.checkpoint_path = os.path.join(base, checkpoint) if checkpoint else None
        self.total_rewards = deque(maxlen=reward_buffer_size)
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.state = self.env.reset()
        self.steps = 0
        self.frame_speed = 0
        self.last_reset_frame = 0
        self.epsilon_start = self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.games = 0

    def create_model(self):
        """
        Create model that will be used for the main and target models.
        Returns:
            Created model
        """
        x0 = Input(self.input_shape)
        x = Conv2D(32, 8, 4, activation='relu')(x0)
        x = Conv2D(64, 4, 2, activation='relu')(x)
        x = Conv2D(64, 3, 1, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, 'relu')(x)
        x = Dense(self.env.action_space.n)(x)
        return Model(x0, x)

    def get_action(self, training=True):
        """
        Generate action following an epsilon-greedy policy.
        Args:
            training: If False, no use of randomness will apply.

        Returns:
            A random action or Q argmax.
        """
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.main_model.predict(np.expand_dims(self.state, 0), verbose=0)
        return np.argmax(q_values)

    def get_buffer_sample(self):
        """
        Get a sample of the replay buffer.
        Returns:
            A batch of observations in the form of
            [[states], [actions], [rewards], [dones], [next states]]
        """
        indices = np.random.choice(self.buffer_size, self.batch_size, replace=False)
        memories = [self.buffer[i] for i in indices]
        batch = [np.array(item) for item in zip(*memories)]
        return batch

    def update(self, batch, gamma):
        """
        Update gradients on a given a batch.
        Args:
            batch: A batch of observations in the form of
                [[states], [actions], [rewards], [dones], [next states]]
            gamma: Discount factor.

        Returns:
            None
        """
        states, actions, rewards, dones, new_states = batch
        q_states = self.main_model.predict(states, verbose=0)
        new_state_values = self.target_model.predict(new_states, verbose=0).max(1)
        new_state_values[dones] = 0
        target_values = np.copy(q_states)
        target_values[np.arange(self.batch_size), actions] = (
            new_state_values * gamma + rewards
        )
        self.main_model.fit(states, target_values, verbose=0)

    def checkpoint(self):
        """
        Save model weights if improved.
        Returns:
            None
        """
        if self.best_reward < self.mean_reward:
            print(f'Best reward updated: {self.best_reward} -> {self.mean_reward}')
            if self.checkpoint_path:
                self.main_model.save_weights(self.checkpoint_path)
        self.best_reward = max(self.mean_reward, self.best_reward)

    def display_metrics(self):
        """
        Display progress metrics to the console.
        Returns:
            None
        """
        display_titles = (
            'frame',
            'games',
            'mean reward',
            'best reward',
            'epsilon',
            'speed',
        )
        display_values = (
            self.steps,
            self.games,
            self.mean_reward,
            self.best_reward,
            np.around(self.epsilon, 3),
            f'{np.around(self.frame_speed, 3)} steps/s',
        )
        display = (
            f'{title}: {value}' for title, value in zip(display_titles, display_values)
        )
        print(strftime('%H:%M:%S', localtime()) + ' | ' + ', '.join(display))

    def update_metrics(self, episode_reward, start_time):
        """
        Update progress metrics.
        Args:
            episode_reward: Total reward per a single episode (game).
            start_time: Episode start time, used for calculating fps.

        Returns:
            None
        """
        self.games += 1
        self.checkpoint()
        self.total_rewards.append(episode_reward)
        self.frame_speed = (self.steps - self.last_reset_frame) / (
            perf_counter() - start_time
        )
        self.last_reset_frame = self.steps
        self.mean_reward = np.around(np.mean(self.total_rewards), 2)
        self.display_metrics()

    def fit(
        self, decay_n_steps=150000,
        learning_rate=1e-4, gamma=0.99,
        update_target_steps=1000,
        weights=r'saves/pong.tf',
        max_steps=None, target_reward=5,
    ):
        """
        Train agent on a supported environment
        Args:
            decay_n_steps: Maximum steps that determine epsilon decay rate.
            learning_rate: Model learning rate shared by both main and target networks.
            gamma: Discount factor used for gradient updates.
            update_target_steps: Update target model every n steps.
            weights: Path to .tf trained model weights to continue training.
            max_steps: Maximum number of steps, if reached the training will stop.
            target_reward: Target reward, if achieved, the training will stop

        Returns:
            None
        """
        episode_reward = 0
        start_time = perf_counter()
        optimizer = Adam(learning_rate)
        if weights and os.path.exists(os.path.join(base, weights)):
            self.main_model.load_weights(os.path.join(base, weights))
            self.target_model.load_weights(os.path.join(base, weights))
        self.main_model.compile(optimizer, loss='mse')
        self.target_model.compile(optimizer, loss='mse')
        while True:
            self.steps += 1
            self.epsilon = max(
                self.epsilon_end, self.epsilon_start - self.steps / decay_n_steps
            )
            action = self.get_action()
            new_state, reward, terminated, truncated = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            self.buffer.append((self.state, action, reward, done, new_state))
            if len(self.buffer) == self.buffer_size:
                self.buffer = []
            self.state = new_state
            if done:
                if self.mean_reward >= target_reward:
                    print(f'Reward achieved in {self.steps} steps!')
                    break
                if max_steps and self.steps >= max_steps:
                    print(f'Maximum steps exceeded')
                    break
                self.update_metrics(episode_reward, start_time)
                start_time = perf_counter()
                episode_reward = 0
                self.state = self.env.reset()
            if len(self.buffer) < self.buffer_size:
                continue
            batch = self.get_buffer_sample()
            self.update(batch, gamma)
            if self.steps % update_target_steps == 0:
                self.target_model.set_weights(self.main_model.get_weights())

    def play(self, env, weights=None):
        """
        Play and display a game.
        Args:
            weights: Path to trained weights, if not specified, the most recent
                model weights will be used.

        Returns:
            None
        """
        if weights:
            self.main_model.load_weights(os.path.join(base, weights))

        self.env = gym.make(env, full_action_space=False, render_mode='human')

        self.state, _ = self.env.reset()
        steps = 0
        
        while True:
            action = self.get_action(False)
            self.state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if done:
                break
            steps += 1


if __name__ == '__main__':
    agn = DQNAgent('ALE/Pong-v5')
    agn.fit()
    agn.play('ALE/Pong-v5', r'saves/pong.tf')