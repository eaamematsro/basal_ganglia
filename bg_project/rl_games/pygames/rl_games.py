import abc
import pdb
import random
from itertools import product
from typing import Dict, Sequence, Tuple

import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np
import pygame
from scipy.ndimage import gaussian_filter1d


class WallObj:
    def __init__(self, x_start: int = 0, x_end: int = 100, y_start: int = 0, y_end: int = 5):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end

        self.width = np.abs(x_end - x_start)
        self.height = np.abs(y_end - y_start)


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
        self.render_mode = None
        self.font = pygame.font.SysFont("Arial_bold", 200)
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
    def __init__(self, rotation: float = 0, max_time_steps: int = 500, testing_mode: bool = False,
                 boundary_size: int = 5, agent_size: int = 8, image_obs: bool = False,
                 **kwargs):
        super().__init__(title=type(self).__name__, **kwargs)

        self.walls = []

        # Add boundaries #
        self.add_wall(x_range=(0, self.width), y_range=(0, boundary_size))
        self.add_wall(x_range=(0, self.width), y_range=(self.height - boundary_size, self.height))
        self.add_wall(x_range=(0, boundary_size), y_range=(0, self.height))
        self.add_wall(x_range=(self.width - boundary_size, self.width), y_range=(0, self.height))

        self.testing = testing_mode
        self.target_pos = None
        self.agent_pos = None
        self.velocity = None
        self.done = False
        self.reward = None
        self.prev_dist = None
        self.score = 0
        self.time_steps = 0
        self.max_time = max_time_steps
        self.betas = 1 * np.eye(2) + 1 * np.asarray(
            [
                [np.cos(rotation * np.pi / 180), -np.sin(rotation * np.pi / 180)],
                [np.sin(rotation * np.pi / 180), np.cos(rotation * np.pi / 180)],
            ]
        )
        self.gain = 1
        self.drifts = np.zeros(2)
        self.agent_size = agent_size
        self.image_obs = image_obs

    def add_wall(self, x_range: Tuple[int, int], y_range: Tuple[int, int]) -> None:
        wall = WallObj(x_start=x_range[0], x_end=x_range[1],
                       y_start=y_range[0], y_end=y_range[1])
        self.walls.append(wall)

    def reset(self):
        """"""
        self.target_pos = None
        self.agent_pos = None
        self.velocity = None
        self.done = False
        self.reward = 0
        self.score = 0
        self.prev_dist = None
        self.time_steps = 0
        self.generate_agent()
        self.generate_target()

    def observe(self):

        if self.image_obs:
            obs = np.array(pygame.surfarray.pixels3d(self.display))
        else:
            obs = np.array(
                [
                    self.agent_pos[0] / self.width - 0.5,
                    self.target_pos[0] / self.width - 0.5,
                    self.agent_pos[1] / self.height - 0.5,
                    self.target_pos[1] / self.height - 0.5,
                ],
                dtype=np.float32,
            )

        return obs

    def generate_target(self):
        if self.target_pos is None or self.done:
            x = random.randrange(3, self.width - 1)
            y = random.randrange(3, self.height - 1)

            while self.detect_wall_collision((x, y)):
                x = random.randrange(3, self.width - 1)
                y = random.randrange(3, self.height - 1)

            self.target_pos = np.array([x, y])

    def generate_agent(self):
        if self.target_pos is None:
            x = random.randrange(3, self.width - 1)
            y = random.randrange(3, self.height - 1)

            while self.detect_wall_collision((x, y)):
                x = random.randrange(3, self.width - 1)
                y = random.randrange(3, self.height - 1)

            self.agent_pos = np.array([x, y])
            self.velocity = (0, 0)

    def detect_collision(self):
        if (
                np.sqrt(
                    ((np.asarray(self.target_pos) - np.asarray(self.agent_pos)) ** 2).sum()
                )
                <= 2 * self.agent_size
        ):
            self.done = True
            self.score += 1
        else:
            self.done = False

    def detect_wall_collision(self, pos):
        for wall in self.walls:
            x_lines = np.arange(wall.x_start, wall.x_end)
            y_lines = np.arange(wall.y_start, wall.y_end)
            xx, yy = np.meshgrid(x_lines, y_lines)
            agent_distances = np.sqrt((pos[0] - xx) ** 2 + (pos[1] - yy) ** 2)

            if agent_distances.min() <= self.agent_size:
                return True

        return False

    def act(self, action: np.ndarray):
        self.time_steps += 1
        if self.done:
            self.reset()

        else:
            x, y = self.agent_pos

            pos = (
                    np.asarray([x, y])
                    + self.gain * (self.betas @ action)
                    + np.asarray(self.drifts)
            )
            wall_collision = self.detect_wall_collision(pos)
            if not wall_collision:
                self.agent_pos = pos

        self.detect_collision()

    def run(self, scale: float = 10):
        self.running = True
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
                        action = np.array([0, -scale])
                    elif event.key == pygame.K_DOWN:
                        action = np.array([0, scale])
                    elif event.key == pygame.K_LEFT:
                        action = np.array([-scale, 0])
                    elif event.key == pygame.K_RIGHT:
                        action = np.array([scale, 0])
            self.act(action)
            self.detect_collision()
            self.evaluate()
            self.view()  # update display surface
        self.close()

    def get_info(
            self,
    ) -> Dict:

        info = {"r": self.reward}
        return info

    def view(self):
        # Draw target and agent icons
        self.display.fill((67, 70, 75))

        img = self.font.render(f"{self.reward: 0.2f}", True, (57, 60, 65))

        self.display.blit(
            img, img.get_rect(center=(20 * self.agent_size + self.agent_size,
                                      self.agent_size * self.agent_size + self.agent_size)).topleft
        )

        pygame.draw.circle(
            self.display,
            "RED",
            (self.target_pos[0], self.target_pos[1]),
            self.agent_size),

        pygame.draw.circle(
            self.display,
            "White",
            (self.agent_pos[0], self.agent_pos[1]),
            self.agent_size
        )

        for wall in self.walls:
            pygame.draw.rect(
                self.display,
                "Black",
                (wall.x_start, wall.y_start, wall.width, wall.height),
            )

        if self.testing:
            pygame.display.update()

    def is_done(
            self,
    ):
        if self.time_steps == self.max_time:
            self.done = True
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


class DegenerateGridWorld(GridWorld):
    def __init__(self, action_dim: int = 10, **kwargs):
        super().__init__(**kwargs)

        transformation_mat = np.random.randn(2, action_dim) / np.sqrt(action_dim)
        self.transition_mat = transformation_mat

    def act(self, action: np.ndarray):
        self.time_steps += 1
        if self.done:
            self.reset()
        else:
            x, y = self.agent_pos

            pos = (
                    np.asarray([x, y])
                    + self.gain * (self.betas @ self.transition_mat @ action)
                    + np.asarray(self.drifts)
            )
            wall_collision = self.detect_wall_collision(pos)
            if not wall_collision:
                self.agent_pos = pos

        self.detect_collision()


class MultiRoomGridWorld(GridWorld):
    def __init__(self, x_divisions: int = 2, y_divisions: int = 2, boundary: int = 5,
                 gap_multiplier: int = 4, **kwargs):
        super().__init__(**kwargs)

        self.x_divs = x_divisions
        if x_divisions > 1:
            self.x_bins = (np.linspace(0, self.width, x_divisions + 1)[:x_divisions]).tolist()
        else:
            self.x_bins = None

        self.y_divs = y_divisions
        if y_divisions > 1:
            self.y_bins = (np.linspace(0, self.height, y_divisions + 1)[:y_divisions]).tolist()
        else:
            self.y_bins = None

        if self.x_bins is not None:
            x_bins = self.x_bins.copy()
            x_bins.append(self.width)
        else:
            x_bins = [0]
            x_bins.append(self.width)

        if self.y_bins is not None:
            y_bins = self.y_bins.copy()
            y_bins.append(self.height)
        else:
            y_bins = [0]
            y_bins.append(self.height)

        for x_div, y_div in product(range(x_divisions), range(y_divisions)):
            if x_div <= x_divisions - 1:
                self.add_wall(
                    x_range=(x_bins[x_div],
                             x_bins[x_div] + int((x_bins[x_div + 1] - x_bins[x_div]) / 2) - gap_multiplier * boundary),
                    y_range=(y_bins[y_div] - boundary, y_bins[y_div] + boundary)
                )

                self.add_wall(
                    x_range=(x_bins[x_div] + int((x_bins[x_div + 1] - x_bins[x_div]) / 2) + gap_multiplier * boundary,
                             x_bins[x_div + 1]),
                    y_range=(y_bins[y_div] - boundary, y_bins[y_div] + boundary)
                )

            if y_div <= y_divisions - 1:
                self.add_wall(
                    y_range=(
                        y_bins[y_div],
                        y_bins[y_div] + int((y_bins[y_div + 1] - y_bins[y_div]) / 2) - gap_multiplier * boundary),
                    x_range=(x_bins[x_div] - boundary, x_bins[x_div] + boundary)
                )

                self.add_wall(
                    y_range=(
                        y_bins[y_div] + int((y_bins[y_div + 1] - y_bins[y_div]) / 2) + gap_multiplier * boundary,
                        y_bins[y_div + 1]),
                    x_range=(x_bins[x_div] - boundary, x_bins[x_div] + boundary)
                )


class MultiWorldGridWorld(GridWorld):
    def __init__(self, x_divisions: int = 2, y_divisions: int = 2, boundary: int = 5,
                 gap_multiplier: int = 4, **kwargs):
        super().__init__(**kwargs)

        self.betas = {}
        self.x_divs = x_divisions
        if x_divisions > 1:
            self.x_bins = (np.linspace(0, self.width, x_divisions + 1)[:x_divisions]).tolist()
        else:
            self.x_bins = None

        self.y_divs = y_divisions
        if y_divisions > 1:
            self.y_bins = (np.linspace(0, self.height, y_divisions + 1)[:y_divisions]).tolist()
        else:
            self.y_bins = None

        if self.x_bins is not None:
            x_bins = self.x_bins.copy()
            x_bins.append(self.width)
        else:
            x_bins = [0]
            x_bins.append(self.width)

        if self.y_bins is not None:
            y_bins = self.y_bins.copy()
            y_bins.append(self.height)
        else:
            y_bins = [0]
            y_bins.append(self.height)

        for x_div, y_div in product(range(x_divisions), range(y_divisions)):
            rotation = np.sqrt(180) * np.random.randn()
            if x_div <= x_divisions - 1:
                self.add_wall(
                    x_range=(x_bins[x_div],
                             x_bins[x_div] + int((x_bins[x_div + 1] - x_bins[x_div]) / 2) - gap_multiplier * boundary),
                    y_range=(y_bins[y_div] - boundary, y_bins[y_div] + boundary)
                )

                self.add_wall(
                    x_range=(x_bins[x_div] + int((x_bins[x_div + 1] - x_bins[x_div]) / 2) + gap_multiplier * boundary,
                             x_bins[x_div + 1]),
                    y_range=(y_bins[y_div] - boundary, y_bins[y_div] + boundary)
                )

            if y_div <= y_divisions - 1:
                self.add_wall(
                    y_range=(
                        y_bins[y_div],
                        y_bins[y_div] + int((y_bins[y_div + 1] - y_bins[y_div]) / 2) - gap_multiplier * boundary),
                    x_range=(x_bins[x_div] - boundary, x_bins[x_div] + boundary)
                )

                self.add_wall(
                    y_range=(
                        y_bins[y_div] + int((y_bins[y_div + 1] - y_bins[y_div]) / 2) + gap_multiplier * boundary,
                        y_bins[y_div + 1]),
                    x_range=(x_bins[x_div] - boundary, x_bins[x_div] + boundary)
                )

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

            wall_collision = self.detect_wall_collision(pos)
            if not wall_collision:
                self.agent_pos = pos


class SineGeneration(PyGame):
    def __init__(
            self,
            amplitudes: Sequence[float] = (1,),
            phases: Sequence[float] = (0,),
            frequencies: Sequence[float] = (1,),
            go_bounds: Sequence[float] = (0.05, 0.2),
            stop_bounds: Sequence[float] = (0.75, 0.9),
            hold_weight: float = 1,
            duration: int = 5,
            step_size: float = 0.01,
            tolerance: float = 0.25,
            **kwargs,
    ) -> None:
        super().__init__(title=type(self).__name__, **kwargs)

        self.episode_length = int(duration / step_size)
        self.step_size = step_size
        self.tolerance = tolerance
        self.hold_weight = hold_weight

        # Store sine parameters
        self.amplitudes = amplitudes
        self.phases = phases
        self.frequencies = frequencies

        # store timing parameters
        self.go_bounds = go_bounds
        self.stop_bounds = stop_bounds

        # intialize state variables
        self.font = pygame.font.SysFont("Arial_bold", 50)

        self.target_position = None
        self.go_cue = None
        self.stop_cue = None
        self.go_time = None
        self.stop_time = None
        self.agent_pos = None
        self.agent_tracker = []
        self.amplitude = None
        self.phase = None
        self.frequency = None
        self.done = False
        self.reward = None
        self.score = 0
        self.time_step = 0
        my_dpi = 100
        self.fig, self.ax = plt.subplots(
            figsize=(self.width / my_dpi, self.height / my_dpi), dpi=my_dpi
        )
        self.reset()

    def observe(self) -> np.ndarray:

        obs = np.array(
            [
                self.go_cue[self.time_step],
                self.stop_cue[self.time_step],
                self.amplitude / np.max(np.abs(self.amplitudes)),
                self.frequency / np.max(np.abs(self.frequencies)),
                self.phase / np.maximum(np.max(np.abs(self.phases)), 1),
            ],
            dtype=np.float32,
        )
        return obs

    def generate_target(self):

        target = np.zeros(self.episode_length)
        go_cue = np.zeros(self.episode_length)
        stop_cue = np.zeros(self.episode_length)

        # get go and stop cue times

        go_time = np.random.randint(
            int(self.go_bounds[0] * self.episode_length),
            int(self.go_bounds[1] * self.episode_length),
        )

        go_cue[go_time - 5: go_time + 5] = 1
        go_cue = gaussian_filter1d(go_cue, sigma=5)
        go_cue /= go_cue.std()

        stop_time = np.random.randint(
            int(self.stop_bounds[0] * self.episode_length),
            int(self.stop_bounds[1] * self.episode_length),
        )

        stop_cue[stop_time - 5: stop_time + 5] = 1
        stop_cue = gaussian_filter1d(stop_cue, sigma=5)
        stop_cue /= stop_cue.std()

        # get sine parameters
        amplitude = np.random.choice(self.amplitudes)
        frequency = np.random.choice(self.frequencies)
        phase = np.random.choice(self.phases)

        times = np.arange(self.episode_length - go_time) * self.step_size

        target[go_time:] = amplitude * np.sin(frequency * 2 * np.pi * (times + phase))
        target[stop_time:] = 0
        self.target_position = target
        self.stop_cue = stop_cue
        self.go_cue = go_cue
        self.go_time = go_time
        self.stop_time = stop_time
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def generate_agent(self):
        self.agent_pos = 0

    def act(self, action):
        self.time_step += 1
        if self.done:
            self.reset()
        else:
            self.agent_pos = action
        self.agent_tracker.append(self.agent_pos)
        self.is_done()

    def evaluate(self):
        if (self.time_step <= self.go_time) or (self.time_step >= self.stop_time):
            weight = self.hold_weight
        else:
            weight = 1

        error = weight * (self.agent_pos - self.target_position[self.time_step]) ** 2
        reward = np.exp(-1 * (error / (self.tolerance ** 2)))

        if type(reward) is float:
            self.reward = reward / self.episode_length
        else:
            self.reward = reward[0] / self.episode_length

        self.score += self.reward

        return self.reward

    def is_done(self):
        if self.time_step >= self.episode_length - 1:
            self.done = True
        return self.done

    def reset(self):
        self.target_position = None
        self.go_cue = None
        self.stop_cue = None
        self.go_time = None
        self.stop_time = None
        self.agent_pos = None
        self.agent_tracker = []
        self.amplitude = None
        self.phase = None
        self.frequency = None
        self.done = False
        self.reward = None
        self.score = 0
        self.time_step = 0

        self.generate_target()
        self.generate_agent()

    def view(self):
        # Draw target and agent icons
        self.ax.cla()
        times = np.arange(self.episode_length) * self.step_size
        self.ax.plot(
            times, self.target_position, label="Target", ls="--", lw=2, color="k"
        )
        self.ax.plot(
            times[: len(self.agent_tracker)],
            self.agent_tracker,
            label="Agent Location",
        )
        self.ax.plot(times, self.go_cue, label="Start Cue", color="g")
        self.ax.plot(times, self.stop_cue, label="Stop Cue", color="r")
        self.ax.legend()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("AU")
        # self.display.fill((67, 70, 75))

        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        self.display.blit(surf, (0, 0))
        img = self.font.render(f"{self.score: 0.2f}", True, (57, 60, 65))

        self.display.blit(
            img, img.get_rect(center=(20 * 15 + 15, 15 * 15 + 15)).topleft
        )
        # pygame.display.flip()

    def get_info(
            self,
    ) -> Dict:
        info = {"r": self.reward}
        return info

    def run(self) -> None:
        self.running = True

        while self.running:
            action = np.random.randn()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        break
            self.act(action)
            self.evaluate()
            self.view()  # update display surface
        self.close()


if __name__ == "__main__":
    game = DegenerateGridWorld(testing_mode=True)
    game.run()
    pass
