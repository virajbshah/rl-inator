import numpy as np
import pygame

import gym
from gym import spaces


class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 24}

    def __init__(self, render_mode=None, bounds=(5, 5)):
        self.bounds = bounds
        self.pix_square_size = 128
        self.window_size = tuple(dim * self.pix_square_size for dim in bounds)

        self.observation_space = spaces.Dict(
            {
                'agent': spaces.Box(np.zeros(2), np.array(bounds), shape=(2,), dtype=np.int32),
                'key': spaces.Box(np.zeros(2), np.array(bounds), shape=(2,), dtype=np.int32),
                'chest': spaces.Box(np.zeros(2), np.array(bounds), shape=(2,), dtype=np.int32),
                'has_key': spaces.Discrete(2),
            }
        )

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            'agent': self._agent_location,
            'key': self._key_location,
            'chest': self._chest_location,
            'has_key': self._has_key,
        }

    def _get_info(self):
        return {
            'key_distance': np.linalg.norm(
                self._agent_location - self._key_location, ord=1
            ),
            'chest_distance': np.linalg.norm(
                self._agent_location - self._chest_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(np.zeros(2), np.array(self.bounds))

        self._key_location = self._agent_location
        while np.array_equal(self._key_location, self._agent_location):
            self._key_location = self.np_random.integers(
                np.zeros(2), np.array(self.bounds)
            )

        self._chest_location = self._agent_location
        while np.array_equal(self._chest_location, self._agent_location) or np.array_equal(self._chest_location, self._key_location):
            self._chest_location = self.np_random.integers(
                np.zeros(2), np.array(self.bounds)
            )

        self._has_key = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        self._agent_location = np.clip(
            self._agent_location + direction, np.zeros(2), np.array(self.bounds) - 1
        )

        if np.array_equal(self._agent_location, self._key_location):
            self._has_key = 1

        terminated = np.array_equal(self._agent_location, self._chest_location) and self._has_key

        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(
                self.pix_square_size * self._key_location,
                (self.pix_square_size, self.pix_square_size),
            ),
        )

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self.pix_square_size * self._chest_location,
                (self.pix_square_size, self.pix_square_size),
            ),
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * self.pix_square_size,
            self.pix_square_size / 3,
        )


        for i in range(self.bounds[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, self.pix_square_size * i),
                (self.window_size[1], self.pix_square_size * i),
                width=3,
            )

        for i in range(self.bounds[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size * i, 0),
                (self.pix_square_size * i, self.window_size[0]),
                width=3,
            )

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == '__main__':
    env = GridWorldEnv(render_mode='human')
    observation, info = env.reset()

    for _ in range(160):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()