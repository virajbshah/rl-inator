from collections import deque

import cv2
import gym
import numpy as np

from gym.spaces import Box


class AtariPreprocessor(gym.Wrapper):
    """
    gym wrapper for preprocessing atari frames.
    """

    def __init__(self, env, frame_skips=4, resize_shape=(84, 84), state_buffer_size=2):
        """
        Initialize preprocessing settings.
        Args:
            env: gym environment that returns states as atari frames.
            frame_skips: Number of frame skips to use per environment step.
            resize_shape: (m, n) output frame size.
            state_buffer_size: State buffer for max pooling.
        """
        super(AtariPreprocessor, self).__init__(env)
        self.skips = frame_skips
        self.frame_shape = resize_shape
        self.observation_space = Box(low=0, high=255, shape=(*resize_shape, 1), dtype=np.uint8)
        self.observation_buffer = deque(maxlen=state_buffer_size)

    def process_frame(self, frame):
        """
        Resize and convert atari frame to grayscale.
        Args:
            frame: Image as numpy.ndarray

        Returns:
            Processed frame.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, self.frame_shape) / 255
        return np.expand_dims(frame, -1)

    def step(self, action: int):
        """
        Step respective to self.skips.
        Args:
            action: Action supported by self.env

        Returns:
            (state, reward, done, info)
        """
        total_reward = 0
        state, terminated, truncated, info = 4 * [None]
        for _ in range(self.skips):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            self.observation_buffer.append(state)
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self.observation_buffer), axis=0)
        return self.process_frame(max_frame), total_reward, terminated or truncated, info

    def reset(self, **kwargs):
        """
        Reset self.env
        Args:
            **kwargs: kwargs passed to self.env.reset()

        Returns:
            Processed atari frame.
        """
        self.observation_buffer.clear()
        observation, _ = self.env.reset(**kwargs)
        self.observation_buffer.append(observation)
        return self.process_frame(observation)