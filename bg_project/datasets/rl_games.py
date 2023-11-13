import pygame
import random
import abc
from abc import ABC


class PyGame(metaclass=abc.ABCMeta):
    def __init__(
        self,
        width: int = 630,
        height: int = 480,
        title: str = "BaseModel",
        fps: int = 10,
        **kwargs
    ):
        pygame.init()
        self.display = pygame.display.set_mode(size=(width, height))
        pygame.display.set_caption(title)
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.width, self.height = width, height

    def close(self):
        pygame.quit()


class SnakeEater(PyGame):
    def __init__(self, rows: int = 30, cols: int = 40, **kwargs):
        super().__init__(title="SnakeEater", **kwargs)
        self.running = False
        self.snake_dir = ""
        self.snake_list = []
        self.apple_pos = []
        self.snake_eat = False
        self.snake_dead = False
        self.score = 0
        self.col = cols
        self.row = rows

    def generate_snake(self):
        if len(self.snake_list) == 0:
            x = random.randrange(3, self.col - 1)
            y = random.randrange(3, self.row - 1)
            self.snake_list.append((x, y))

            # body
            self.snake_list.append(
                random.choice([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
            )

            # tail
            x, y = self.snake_list[-1]
            temp = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            temp.remove(self.snake_list[0])
            self.snake_list.append(random.choice(temp))

        if len(self.snake_dir) == 0:
            dir_list = ["up", "down", "left", "right"]
            if self.snake_list[0][0] > self.snake_list[1][0]:
                dir_list.remove("left")
            elif self.snake_list[0][0] < self.snake_list[1][0]:
                dir_list.remove("right")

            if self.snake_list[0][1] > self.snake_list[1][1]:
                dir_list.remove("up")
            elif self.snake_list[0][1] < self.snake_list[1][1]:
                dir_list.remove("down")

            self.snake_dir = random.choice(dir_list)

    def generate_apple(self):
        if len(self.apple_pos) == 0:
            x = random.randrange(3, self.col - 1)
            y = random.randrange(3, self.row - 1)

            while (x, y) in self.snake_list:
                x = random.randrange(3, self.col - 1)
                y = random.randrange(3, self.row - 1)

            self.apple_pos = (x, y)

    def update_snake(self):
        if not self.snake_dead:
            if not self.snake_eat:
                self.snake_list.pop(-1)
            else:
                self.snake_eat = False

            if self.snake_dir == "up":
                self.snake_list.insert(
                    0, (self.snake_list[0][0], self.snake_list[0][1] - 1)
                )
            elif self.snake_dir == "down":
                self.snake_list.insert(
                    0, (self.snake_list[0][0], self.snake_list[0][1] + 1)
                )
            elif self.snake_dir == "left":
                self.snake_list.insert(
                    0, (self.snake_list[0][0] - 1, self.snake_list[0][1])
                )
            elif self.snake_dir == "right":
                self.snake_list.insert(
                    0, (self.snake_list[0][0] + 1, self.snake_list[0][1])
                )

    def collision(self):
        if self.snake_list[0] == self.apple_pos:
            self.snake_eat = True
            self.score += 1
            self.apple_pos = []

        if self.snake_list[0][1] == 1 and self.snake_dir == "up":
            self.snake_dead = True
        elif self.snake_list[0][1] == self.row - 1 and self.snake_dir == "down":
            self.snake_dead = True
        elif self.snake_list[0][0] == 1 and self.snake_dir == "left":
            self.snake_dead = True
        elif self.snake_list[0][0] == self.col - 1 and self.snake_dir == "right":
            self.snake_dead = True

        if self.snake_list[0] in self.snake_list[1:]:
            self.snake_dead = True

    def run(self, boundary_thickness: int = 15):
        self.running = True
        width, height = self.width, self.height
        font = pygame.font.SysFont("Arial_bold", 380)

        while self.running:
            self.generate_snake()
            self.update_snake()
            self.collision()
            self.generate_apple()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        break

                    if (
                        event.key == pygame.K_UP
                        and not self.snake_list[0][1] > self.snake_list[1][1]
                    ):
                        self.snake_dir = "up"
                    elif (
                        event.key == pygame.K_DOWN
                        and not self.snake_list[0][1] < self.snake_list[1][1]
                    ):
                        self.snake_dir = "down"
                    elif (
                        event.key == pygame.K_LEFT
                        and not self.snake_list[0][0] > self.snake_list[1][0]
                    ):
                        self.snake_dir = "left"
                    elif (
                        event.key == pygame.K_RIGHT
                        and not self.snake_list[0][0] < self.snake_list[1][0]
                    ):
                        self.snake_dir = "right"

            # draw on screen
            self.display.fill((67, 70, 75))
            pygame.draw.rect(
                self.display, "WHITE", (0, 0, width, boundary_thickness)
            )
            pygame.draw.rect(
                self.display, "WHITE", (0, 0, boundary_thickness, height)
            )
            pygame.draw.rect(
                self.display,
                "WHITE",
                (0, height - boundary_thickness, width, boundary_thickness),
            )
            pygame.draw.rect(
                self.display,
                "WHITE",
                (width - boundary_thickness, 0, boundary_thickness, height),
            )

            if self.snake_dead:
                img = font.render(str(self.score), True, (125, 85, 85))
            else:
                img = font.render(str(self.score), True, (57, 60, 65))

            self.display.blit(
                img, img.get_rect(center=(20 * 15 + 15, 15 * 15 + 15)).topleft
            )

            if len(self.apple_pos) > 0:
                pygame.draw.rect(
                    self.display,
                    "RED",
                    (self.apple_pos[0] * 15, self.apple_pos[1] * 15, 15, 15),
                )

            for body_part in self.snake_list[1:]:
                pygame.draw.rect(
                    self.display,
                    (180, 180, 180),
                    (body_part[0] * 15, body_part[1] * 15, 15, 15),
                )

            pygame.draw.rect(
                self.display,
                "WHITE",
                (self.snake_list[0][0] * 15, self.snake_list[0][1] * 15, 15, 15),
            )

            pygame.display.update()  # update display surface
            self.clock.tick(self.fps)

        self.close()
