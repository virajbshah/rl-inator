import numpy as np
import pygame

import gym
from gym import spaces


class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 12}

    def __init__(self, render_mode=None, bounds=(50, 50)):
        self.bounds = bounds
        self.pix_square_size = 16
        self.window_size = tuple(dim * self.pix_square_size for dim in bounds)

        self._origin_location = np.array((bounds)) // 2
        self._length_step_size = 1
        self._angle_step_size = 0.1

        self.observation_space = spaces.Dict(
            {
                'pick': spaces.Box(np.zeros(2), np.array(bounds), shape=(2,), dtype=np.int32),
                'place': spaces.Box(np.zeros(2), np.array(bounds), shape=(2,), dtype=np.int32),
                'agent': spaces.Box(
                    np.array([0, 0]), np.array(np.sqrt([bounds[0] ** 2 + bounds[1] ** 2, 2 * np.pi])), shape=(2,), dtype=np.float64
                ),
                'has_obj': spaces.Discrete(2),
            }
        )

        self.action_space = spaces.MultiDiscrete([3, 3], dtype=np.int32)

        self._action_to_direction = {

        }

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            'agent': np.array([self._agent_length, self._agent_angle]),
            'pick': self._pick_location,
            'place': self._place_location,
            'has_obj': self._has_obj,
        }

    def _get_info(self):
        return {
            'pick_distance': np.linalg.norm(
                self._get_agent_location() - self._pick_location, ord=1
            ),
            'place_distance': np.linalg.norm(
                self._get_agent_location() - self._place_location, ord=1
            )
        }

    def _get_agent_location(self):
        return np.array(
            [
                self._origin_location[0] + self._agent_length * np.cos(self._agent_angle),
                self._origin_location[1] + self._agent_length * np.sin(self._agent_angle)
            ], dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_length = 0
        self._agent_angle = 0

        self._pick_location = np.zeros(2, dtype=np.int32)
        while np.array_equal(self._pick_location, np.zeros(2, dtype=np.int32)):
            self._pick_location = self.np_random.integers(
                np.zeros(2), np.array(self.bounds)
            )

        self._place_location = np.zeros(2, dtype=np.int32)
        while (np.array_equal(self._place_location, np.zeros(2, dtype=np.int32)) or  
               np.array_equal(self._place_location, self._pick_location)):
            self._place_location = self.np_random.integers(
                np.zeros(2), np.array(self.bounds)
            )

        self._has_obj = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        self._agent_length = np.clip(
            self._agent_length + (action[0] - 1) * self._length_step_size,
            0, np.sqrt(self.bounds[0] ** 2 + self.bounds[1] ** 2)
        )
        self._agent_angle = (self._agent_angle + (action[1] - 1) * self._angle_step_size) % (2 * np.pi)

        if np.array_equal(self._get_agent_location(), self._pick_location):
            self._has_obj = 1

        terminated = np.array_equal(self._get_agent_location(), self._place_location) and self._has_obj

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
                self.pix_square_size * self._pick_location,
                (self.pix_square_size, self.pix_square_size),
            ),
        )

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self.pix_square_size * self._place_location,
                (self.pix_square_size, self.pix_square_size),
            ),
        )

        pygame.draw.line(
            canvas,
            128,
            (np.array(self._origin_location) + 0.5) * self.pix_square_size,
            (self._get_agent_location() + 0.5) * self.pix_square_size,
            width=5
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (self._origin_location + 0.5) * self.pix_square_size,
            self.pix_square_size / 3,
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._get_agent_location() + 0.5) * self.pix_square_size,
            self.pix_square_size / 3,
        )


        # for i in range(self.bounds[0] + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, self.pix_square_size * i),
        #         (self.window_size[1], self.pix_square_size * i),
        #         width=3,
        #     )

        # for i in range(self.bounds[1] + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (self.pix_square_size * i, 0),
        #         (self.pix_square_size * i, self.window_size[0]),
        #         width=3,
        #     )

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
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

        if terminated or truncated:
            observation, info = env.reset()

    env.close()