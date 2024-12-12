import torch
import random
import numpy as np
from collections import deque
# deque to store our memory  
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        # A measure of exploration vs. exploitation. Initially set to 0, meaning the agent will initially exploit more.
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() when we exceed the memeory will call it
        self.model = Linear_QNet(11, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game): #retrieves the current state of the game, which will be used as input for the neural network.
        head = game.snake[0]    
# These points are used to determine if the snake is about to collide with something in those directions.
# The grid in the game is divided into cells of size 20x20, so each point is 20 units away from the head in the respective direction (left, right, up, down).
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
    #    check which direction the snake is currently moving in. For example, if the snake is moving to the left, dir_l will be True, and the other directions (dir_r, dir_u, dir_d) will be False.
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight - checks if there is danger (a collision) directly in front of the snake. 
            # if we are going right and the point right of us gives a collision then gives danger
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        return np.array(state, dtype=int)   # convert boolean t/f into 0 /1
    
    # remember Method: This method stores the experiences (state, action, reward, next_state, done) in the agent's memory.
# It appends the experience tuple to the deque, automatically removing the oldest experience if MAX_MEMORY is exceeded.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

# self.memory is a deque (double-ended queue) that stores the agent's experiences in the form of tuples. Each tuple typically contains:
# The state the agent was in.
# The action the agent took.
# The reward received after taking that action.
# The next state the agent transitioned to.
# A boolean indicating whether the episode has finished (i.e., whether the game has ended).
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # if the number of experiences stored in memory exceeds BATCH_SIZE. The idea here is to sample a subset of experiences for training, which helps in breaking the correlation between consecutive experiences.
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

#  unpacks the tuples in mini_sample into separate lists: states, actions, rewards, next_states, and dones. The zip(*mini_sample) technique is a common Python idiom to unpack lists of tuples. 
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # # Training Step:
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
# The train_step method will process these inputs and update the Q-learning model based on the experiences. This involves:
# Calculating the predicted Q-values for the current states.
# Updating the target Q-values based on the rewards and the predicted Q-values of the next states.
# Calculating the loss between the predicted and target Q-values.
# Using backpropagation to adjust the model’s weights based on this loss, effectively training the model to improve its predictions in the future.
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # This comment explains the purpose of the following code. It highlights the balance between exploration (trying new actions) and exploitation (using known actions that yield high rewards)
        # Epsilon is a parameter used in the epsilon-greedy strategy:
# High epsilon means the agent is more likely to explore (take random actions).
# Low epsilon means the agent is more likely to exploit (take the best-known action based on past experiences).
# As the agent plays more games, epsilon decreases, encouraging it to exploit learned actions more than exploring new ones.    
        self.epsilon = 80 - self.n_games  #set to a value that decreases as the number of games (n_games) increases.
        final_move = [0,0,0] #Index 0: Move straight. Index 1: Turn right. Index 2: Turn left. The list will be modified to indicate which action the agent should take.

        # Random Action Selection (Exploration)
        if random.randint(0, 200) < self.epsilon:
            # If the generated number is less than self.epsilon, the agent will explore (take a random action).
            move = random.randint(0, 2)
            final_move[move] = 1
            # The final_move list is updated to reflect the randomly chosen action. For example, if move is 0, it sets final_move to [1, 0, 0], indicating a straight move.
        else:
            # Predicting Action (Exploitation) - which executes when the agent decides to exploit its knowledge (i.e., take the best-known action) rather than explore new actions.
            state0 = torch.tensor(state, dtype=torch.float) #converts the current state, which is likely a NumPy array or a list, into a PyTorch tensor. specifies that the tensor should have a floating-point type. This is necessary because neural networks typically work with floating-point numbers for calculations.
            prediction = self.model(state0)
            # this will execute the forward function in model.py

            #  the converted tensor state0 is passed through the neural network model (which is an instance of Linear_QNet) to obtain the predicted Q-values for each possible action based on the current state.
# The prediction variable now holds the output from the model, which typically consists of three Q-values corresponding to the three actions: going straight, turning right, and turning left.
            move = torch.argmax(prediction).item()
            # torch.argmax(prediction) to find the index of the action with the highest Q-value in the prediction tensor. This indicates the action that the model predicts will yield the greatest reward.
            # The .item() method converts the resulting tensor (which contains a single element) into a standard Python integer, assigning it to the variable move.
            final_move[move] = 1
            

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()