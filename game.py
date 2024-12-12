import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# reset
# reward
# play(action) - > returns direction
# game_iteration keep track
# is_collision

Point = namedtuple('Point', 'x, y')
#  acts like a normal class, but the objects it creates are immutable (like tuples) and have named fields for accessing data.
# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
    # sets the initial direction of the snake to the right.
        self.head = Point(self.w/2, self.h/2)
        # creates a new Point object representing the head of the snake, positioned at the center of the game area (using half the width and height).

        #  directly to the left of the head - moves the x-coordinate of the head to the left by one block, leaving the y-coordinate the same (self.head.y), so it's aligned horizontally. The result is a new body segment placed exactly one block to the left of the head
        # moves two blocks left of the head, with the y-coordinate unchanged (self.head.y), creating the second body segment.
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        # The snake is represented as a list of Point objects, where each Point represents a specific segment of the snake's body (its head and its following body segments).
        # self.snake is a list of Point objects, representing the snake's body.
        # The head is at the front, and the body segments follow, each spaced by a distance of BLOCK_SIZE.
        # The snake starts out aligned horizontally, with its head on the right and the body trailing to the left.
        self.score = 0
        self.food = None
        self._place_food()  #presumably generates a new food item in the game.
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        # generates a random integer x representing the x-coordinate of the food.
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        #  creates a new Point object at the randomly generated coordinates (x, y) and assigns it to self.food, which represents the position of the food on the game board.
        if self.food in self.snake: 
            # This checks if the newly placed food's position overlaps with any part of the snake's body. If the food's coordinates are the same as any segment of the snake, it means the food has been placed on top of the snake.
            self._place_food()
            # If the food is placed on the snake, the method calls itself recursively to generate a new food position


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        # The snake's length increases automatically when it eats food because the code inserts the new head at the front of the self.snake list, and does not remove the last segment when food is eaten.
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            # The snake's head moves to the new position where the food is located.
# The body grows because the code doesn't remove the tail of the snake when it eats food.
# The food is then replaced (self._place_food()), and the score is incremented (self.score += 1).
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

# Purpose: Check if a given point (pt) or the snake's head has collided with either the game boundaries or the snake's own body.
# Input: pt: A point (typically the snake's head). If no point is provided, it defaults to the current head of the snake.
# is_collision function checks for two types of collisions
# Boundary collision: The snake's head or a given point has crossed the game boundaries.
# Self-collision: The snake's head or a given point has collided with the snake's own body.
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself - if the point (pt) is located anywhere in the snake's body, except the head
        if pt in self.snake[1:]:
            return True

        return False


# The for loop iterates through each segment of the snake (self.snake), where pt represents the position (coordinates) of each segment.

# First Line: Draws each snake segment as a rectangle:
# Second Line: Draws a smaller inner rectangle inside each snake segment for a visual effect: - defines a smaller rectangle positioned slightly inside the original snake segment (by 4 pixels on both the x and y axes), with dimensions 12x12.
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)) # drawing food

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        #  Blits (draws) the text surface on the game display.
        pygame.display.flip()


    def _move(self, action):
        # based on input action we will decide the next move
        # [straight, right, left]
# represents the four possible directions in which the snake can move: right, down, left, and up (in a clockwise order).
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # Current Direction Index:
        idx = clock_wise.index(self.direction)
        # gets the current direction of the snake (self.direction) and finds its index in the clock_wise list. For example, if the snake is currently moving right, idx will be 0

        if np.array_equal(action, [1, 0, 0]): #go straight.
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]): # turn right.
            next_idx = (idx + 1) % 4
            # The direction is updated to the next clockwise direction, i.e., (idx + 1) % 4 ensures circular movement in the list (e.g., from right to down).
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1] -  turn left.
            #  The direction is updated to the previous direction in the clockwise sequence
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)