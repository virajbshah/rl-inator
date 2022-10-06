import numpy as np
import pygame

import gym
from gym import spaces


class SnakeEnv(gym.Env):
    """
    An OpenAI gym environment simulating the classic
    game known as Snake, for reinforcement learning.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 12}

    def __init__(self, render_mode=None, bounds=(20, 15)):
        self.width, self.height = bounds
        self.pix_square_size = 32
        self.window_size = tuple(dim * self.pix_square_size for dim in bounds)

        self.snake_body_color = (255, 255,   0)
        self.snake_head_color = (  0, 255,   0)
        self.target_color     = (255,   0,   0)

        self.observation_space = spaces.Box(low=0, high=255, shape=(*bounds, 3), dtype=np.int32)

        self.action_space = spaces.Discrete(4)
        
        assert render_mode is None or render_mode in SnakeEnv.metadata['render_modes']
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        observation = np.zeros((self.width, self.height, 3), dtype=np.int32)
        
        observation[self._target_location] = np.array(self.target_color)
        for segment in self._snake_body[:-1]:
            observation[segment] = np.array(self.snake_body_color)
        observation[self._snake_body[-1]] = np.array(self.snake_head_color)

        return observation

    def _get_info(self):
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._snake_length = 1
        self._snake_body = [
            self.np_random.integers(
                np.zeros(2,), np.array([self.width - 1, self.height - 1]), dtype=np.int32
            )
        ]

        self._target_location = self._snake_body[-1]
        while np.array_equal(self._target_location, self._snake_body[-1]):
            self._target_location = self.np_random.integers(
                np.zeros(2,), np.array([self.width - 1, self.height - 1]), dtype=np.int32
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        current_head = self._snake_body[-1]
        if action == 0:   # Up
            next_head = np.array([current_head[0], current_head[1] - 1])
        elif action == 1: # Left
            next_head = np.array([current_head[0] - 1, current_head[1]])
        elif action == 2: # Down
            next_head = np.array([current_head[0], current_head[1] + 1])
        elif action == 3: # Right
            next_head = np.array([current_head[0] + 1, current_head[1]])
        else:
            next_head = current_head

        if not np.array_equal(
            next_head, np.clip(next_head, np.zeros((2,), dtype=np.int32), np.array([self.width - 1, self.height - 1]))
        ) or list(next_head) in np.array(self._snake_body).tolist():
            terminated = True
            reward = -1
        else:
            terminated = False

            if np.array_equal(next_head, self._target_location):
                reward = 1
                self._snake_length += 1
            else:
                reward = -0.01

        next_head = np.clip(next_head, np.zeros((2,), dtype=np.int32), np.array([self.width - 1, self.height - 1]))

        self._snake_body.append(next_head)
        if self._snake_length < len(self._snake_body):
            self._snake_body.pop(0)

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
            self.target_color,
            pygame.Rect(
                self.pix_square_size * self._target_location,
                (self.pix_square_size, self.pix_square_size),
            ),
        )

        for segment in self._snake_body[:-1]:
            pygame.draw.rect(
                canvas,
                self.snake_body_color,
                pygame.Rect(
                    self.pix_square_size * segment,
                    (self.pix_square_size, self.pix_square_size),
                )
            )

        pygame.draw.rect(
            canvas,
            self.snake_head_color,
            pygame.Rect(
                self.pix_square_size * self._snake_body[-1],
                (self.pix_square_size, self.pix_square_size)
            )
        )

        # for i in range(self.width + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (self.pix_square_size * i, 0),
        #         (self.pix_square_size * i, self.window_size[1]),
        #         width=3,
        #     )

        # for i in range(self.height + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, self.pix_square_size * i),
        #         (self.window_size[0], self.pix_square_size * i),
        #         width=3,
        #     )

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])
        elif self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == '__main__':
    env = SnakeEnv(render_mode='human')
    observation, info = env.reset()

    for _ in range(160):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()