# import pdb
#
# import pygame
# import random
#
#
# def generate_snake(SnakeList, SnakeDir):
#     if len(SnakeList) == 0:
#         x = random.randrange(3, col - 1)
#         y = random.randrange(3, row - 1)
#         SnakeList.append((x, y))
#
#         # body
#         SnakeList.append(
#             random.choice([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
#         )
#
#         # tail
#         x, y = SnakeList[-1]
#         temp = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
#         temp.remove(SnakeList[0])
#         SnakeList.append(random.choice(temp))
#
#     if len(SnakeDir) == 0:
#         dir_list = ["up", "down", "left", "right"]
#         if SnakeList[0][0] > SnakeList[1][0]:
#             dir_list.remove("left")
#         elif SnakeList[0][0] < SnakeList[1][0]:
#             dir_list.remove("right")
#
#         if SnakeList[0][1] > SnakeList[1][1]:
#             dir_list.remove("up")
#         elif SnakeList[0][1] < SnakeList[1][1]:
#             dir_list.remove("down")
#
#         SnakeDir = random.choice(dir_list)
#
#     return SnakeList, SnakeDir
#
#
# def generate_apple(SnakeList, ApplePos):
#     if len(ApplePos) == 0:
#         x = random.randrange(3, col - 1)
#         y = random.randrange(3, row - 1)
#
#         while (x, y) in SnakeList:
#             x = random.randrange(3, col - 1)
#             y = random.randrange(3, row - 1)
#
#         ApplePos = (x, y)
#
#     return ApplePos
#
#
# def update_snake(SnakeDir, SnakeList, SnakeEat, SnakeDead):
#     if not SnakeDead:
#         if not SnakeEat:
#             SnakeList.pop(-1)
#         else:
#             SnakeEat = False
#
#         if SnakeDir == "up":
#             SnakeList.insert(0, (SnakeList[0][0], SnakeList[0][1] - 1))
#         elif SnakeDir == "down":
#             SnakeList.insert(0, (SnakeList[0][0], SnakeList[0][1] + 1))
#         elif SnakeDir == "left":
#             SnakeList.insert(0, (SnakeList[0][0] - 1, SnakeList[0][1]))
#         elif SnakeDir == "right":
#             SnakeList.insert(0, (SnakeList[0][0] + 1, SnakeList[0][1]))
#
#     return SnakeList, SnakeEat
#
#
# def collision(SnakeList, ApplePos, SnakeDir, SnakeEat, SnakeDead, score):
#     if SnakeList[0] == ApplePos:
#         SnakeEat = True
#         score += 1
#         ApplePos = []
#
#     if snake_list[0][1] == 1 and SnakeDir == "up":
#         SnakeDead = True
#     elif snake_list[0][1] == row - 1 and SnakeDir == "down":
#         SnakeDead = True
#     elif snake_list[0][0] == 1 and SnakeDir == "left":
#         SnakeDead = True
#     elif snake_list[0][0] == col - 1 and SnakeDir == "right":
#         SnakeDead = True
#
#     if SnakeList[0] in SnakeList[1:]:
#         SnakeDead = True
#     return SnakeEat, ApplePos, SnakeDead, score
#
#
# pygame.init()
# width, height = 630, 480
# row, col = 30, 40
# screen = pygame.display.set_mode(size=(width, height))
# pygame.display.set_caption("Runner")
# fps = 10
# boundary_thickness = 5
# clock = pygame.time.Clock()
# font = pygame.font.SysFont("Arial_bold", 380)
# run = True
#
# # Control variables
# snake_dir = ""
# snake_list = []
# apple_pos = []
# snake_eat = False
# snake_dead = False
# score = 0
#
#
# while run:
#     snake_list, snake_dir = generate_snake(SnakeList=snake_list, SnakeDir=snake_dir)
#     snake_list, snake_eat = update_snake(snake_dir, snake_list, snake_eat, snake_dead)
#     snake_eat, apple_pos, snake_dead, score = collision(
#         snake_list, apple_pos, snake_dir, snake_eat, snake_dead, score
#     )
#     apple_pos = generate_apple(snake_list, apple_pos)
#
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             run = False
#             break
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_ESCAPE:
#                 run = False
#                 break
#
#             if event.key == pygame.K_UP and not snake_list[0][1] > snake_list[1][1]:
#                 snake_dir = "up"
#             elif event.key == pygame.K_DOWN and not snake_list[0][1] < snake_list[1][1]:
#                 snake_dir = "down"
#             elif event.key == pygame.K_LEFT and not snake_list[0][0] > snake_list[1][0]:
#                 snake_dir = "left"
#             elif (
#                 event.key == pygame.K_RIGHT and not snake_list[0][0] < snake_list[1][0]
#             ):
#                 snake_dir = "right"
#
#     # draw on screen
#     screen.fill((67, 70, 75))
#     pygame.draw.rect(screen, "WHITE", (0, 0, width, boundary_thickness))
#     pygame.draw.rect(screen, "WHITE", (0, 0, boundary_thickness, height))
#     pygame.draw.rect(
#         screen, "WHITE", (0, height - boundary_thickness, width, boundary_thickness)
#     )
#     pygame.draw.rect(
#         screen, "WHITE", (width - boundary_thickness, 0, boundary_thickness, height)
#     )
#
#     if snake_dead:
#         img = font.render(str(score), True, (125, 85, 85))
#     else:
#         img = font.render(str(score), True, (57, 60, 65))
#
#     screen.blit(img, img.get_rect(center=(20 * 15 + 15, 15 * 15 + 15)).topleft)
#
#     if len(apple_pos) > 0:
#         pygame.draw.rect(screen, "RED", (apple_pos[0] * 15, apple_pos[1] * 15, 15, 15))
#
#     for body_part in snake_list[1:]:
#         pygame.draw.rect(
#             screen, (180, 180, 180), (body_part[0] * 15, body_part[1] * 15, 15, 15)
#         )
#
#     pygame.draw.rect(
#         screen, "WHITE", (snake_list[0][0] * 15, snake_list[0][1] * 15, 15, 15)
#     )
#
#     pygame.display.update()  # update display surface
#     clock.tick(fps)
#
# pygame.quit()
from datasets.rl_games import SnakeEater

game = SnakeEater()

game.run()
