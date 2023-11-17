import pdb

import pygame
import random
import abc
import numpy as np
from itertools import product
from abc import ABC
from typing import Dict


class PyGame(metaclass=abc.ABCMeta):
    def __init__(
        self,
        width: int = 630,
        height: int = 480,
        title: str = "BaseModel",
        fps: int = 10,
        **kwargs,
    ):
        pygame.init()
        self.display = pygame.display.set_mode(size=(width, height))
        pygame.display.set_caption(title)
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.width, self.height = width, height
        self.running = False

    def close(self):
        pygame.quit()

    @abc.abstractmethod
    def reset(self):
        """Resets the game"""

    @abc.abstractmethod
    def observe(self):
        """Returns the current state of the game"""

    @abc.abstractmethod
    def act(self, action):
        """Iterates the game given an action"""

    @abc.abstractmethod
    def evaluate(
        self,
    ):
        """Returns reward of the current frame"""

    @abc.abstractmethod
    def is_done(
        self,
    ):
        """Checks whether game is finished"""

    @abc.abstractmethod
    def get_info(
        self,
    ) -> Dict:
        """Returns info about current episode"""

    @abc.abstractmethod
    def view(self):
        """"""


class GridWorld(PyGame):
    def __init__(self, rotation: float = 90, **kwargs):
        super().__init__(title=type(self).__name__, **kwargs)
        self.target_pos = None
        self.agent_pos = None
        self.velocity = None
        self.done = False
        self.reward = None
        self.prev_dist = None
        self.score = 0
        self.dist = 0
        self.betas = 1 * np.eye(2) + 1 * np.asarray(
            [
                [np.cos(rotation * np.pi / 180), -np.sin(rotation * np.pi / 180)],
                [np.sin(rotation * np.pi / 180), np.cos(rotation * np.pi / 180)],
            ]
        )
        self.gain = 15
        self.drifts = np.zeros(2)

    def reset(self):
        """"""
        self.target_pos = None
        self.agent_pos = None
        self.velocity = None
        self.done = False
        self.reward = None
        self.score = 0
        self.prev_dist = None
        self.generate_agent()
        self.generate_target()

    def observe(self):
        obs = np.array([self.target_pos, self.agent_pos])
        return obs

    def generate_target(self):
        if self.target_pos is None or self.done:
            x = random.randrange(3, self.width - 1)
            y = random.randrange(3, self.height - 1)
            self.target_pos = np.array([x, y])

    def generate_agent(self):
        if self.target_pos is None:
            x = random.randrange(3, self.width - 1)
            y = random.randrange(3, self.height - 1)
            self.agent_pos = np.array([x, y])
            self.velocity = (0, 0)

    def detect_collision(self, boundary: int = 5):
        if (
            np.sqrt(
                ((np.asarray(self.target_pos) - np.asarray(self.agent_pos)) ** 2).sum()
            )
            <= 15
        ):
            self.done = True
            self.score += 1
        else:
            self.done = False

        if self.agent_pos[0] < 0:
            self.agent_pos = (0, self.agent_pos[1])
        elif self.agent_pos[0] > self.width - boundary:
            self.agent_pos = (self.width - boundary, self.agent_pos[1])
        if self.agent_pos[1] < 0:
            self.agent_pos = (self.agent_pos[0], 0)
        elif self.agent_pos[1] > self.height - boundary:
            self.agent_pos = (self.agent_pos[0], self.height - boundary)

    def act(self, action: np.ndarray):

        if self.done:
            self.generate_agent()
            self.generate_target()
            self.done = False

        else:
            x, y = self.agent_pos

            pos = (
                np.asarray([x, y])
                + self.gain * (self.betas @ action)
                + np.asarray(self.drifts)
            )

            self.agent_pos = pos

        self.detect_collision()

    def run(self):
        self.running = True
        font = pygame.font.SysFont("Arial_bold", 200)

        while self.running:
            self.generate_agent()
            self.generate_target()
            action = np.zeros(2)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        break
                    if event.key == pygame.K_UP:
                        action = np.array([0, -1])
                    elif event.key == pygame.K_DOWN:
                        action = np.array([0, 1])
                    elif event.key == pygame.K_LEFT:
                        action = np.array([-1, 0])
                    elif event.key == pygame.K_RIGHT:
                        action = np.array([1, 0])
            self.act(action)
            self.detect_collision()
            self.evaluate()

            # Draw target and agent icons
            self.display.fill((67, 70, 75))

            img = font.render(f"{self.reward: 0.2f}", True, (57, 60, 65))

            self.display.blit(
                img, img.get_rect(center=(20 * 15 + 15, 15 * 15 + 15)).topleft
            )

            pygame.draw.rect(
                self.display,
                "RED",
                (self.target_pos[0], self.target_pos[1], 15, 15),
            )

            pygame.draw.rect(
                self.display,
                "White",
                (self.agent_pos[0], self.agent_pos[1], 15, 15),
            )

            self.view()  # update display surface
        self.close()

    def get_info(
        self,
    ) -> Dict:

        info = {"r": self.reward}
        return info

    def view(self):
        pygame.display.update()
        self.clock.tick(self.fps)

    def is_done(
        self,
    ):
        return self.done

    def evaluate(
        self,
    ):
        """"""
        current_distance = np.sqrt(((self.target_pos - self.agent_pos) ** 2).sum())
        reward_distance = 100 * np.exp(-current_distance / ((0.01 * self.width) ** 2))
        direction_reward = 0
        if self.prev_dist is not None:
            if self.prev_dist > current_distance:
                direction_reward = 1
            elif self.prev_dist < current_distance:
                direction_reward = -5
        self.reward = reward_distance + direction_reward
        self.prev_dist = current_distance
        return self.reward


class MultiWorldGridWorld(GridWorld):
    def __init__(self, x_divisions: int = 2, y_divisions: int = 1, **kwargs):
        super().__init__(**kwargs)

        self.betas = {}
        self.x_divs = x_divisions
        if x_divisions > 1:
            self.x_bins = np.linspace(0, self.width, x_divisions + 1)[:x_divisions]
        else:
            self.x_bins = None

        self.y_divs = y_divisions
        if y_divisions > 1:
            self.y_bins = np.linspace(0, self.height, y_divisions + 1)[:y_divisions]
        else:
            self.y_bins = None

        for x_div, y_div in product(range(x_divisions), range(y_divisions)):
            rotation = np.sqrt(180) * np.random.randn()

            self.betas.update(
                {
                    (x_div, y_div): 0 * np.eye(2)
                    + np.asarray(
                        [
                            [
                                np.cos(rotation * np.pi / 180),
                                -np.sin(rotation * np.pi / 180),
                            ],
                            [
                                np.sin(rotation * np.pi / 180),
                                np.cos(rotation * np.pi / 180),
                            ],
                        ]
                    )
                }
            )

    def act(self, action):

        if self.done:
            self.generate_agent()
            self.generate_target()
            self.done = False

        else:
            x, y = self.agent_pos

            if self.x_bins is not None:
                idx = np.digitize(x, self.x_bins) - 1
            else:
                idx = 0

            if self.y_bins is not None:
                idy = np.digitize(y, self.y_bins) - 1
            else:
                idy = 0
            pos = (
                np.asarray([x, y])
                + self.gain * (self.betas[(idx, idy)] @ action)
                + np.asarray(self.drifts)
            )
            self.agent_pos = (pos[0], pos[1])
