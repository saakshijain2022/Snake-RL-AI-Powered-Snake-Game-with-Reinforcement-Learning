import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# two classes: Linear_QNet, a simple fully-connected neural network used for Q-learning, and 
# QTrainer, which is responsible for training the neural network using Deep Q-Learning.

# nn.Module: The base class for all neural networks in PyTorch
class Linear_QNet(nn.Module):
    # input_size: The number of input features (e.g., 11 in the snake game).
    # hidden_size: The size of the hidden layer (256 in this case).
# output_size: The number of output neurons (3 in the snake game - corresponding to possible actions: [left, straight, right]).
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # connects the input layer to the hidden layer.
        self.linear2 = nn.Linear(hidden_size, output_size) #connects the hidden layer to the output layer.

# defines the forward pass through the network, i.e., how data flows through the network.
    def forward(self, x):
        x = F.relu(self.linear1(x)) #Applies ReLU (Rectified Linear Unit) activation function, introducing non-linearity after the first layer
        x = self.linear2(x) #Passes the output from the hidden layer to the output layer.
        return x #Returns the final output, which represents the Q-values for each action.

# This saves the model's weights (parameters) to a file.
# self.state_dict(): Returns a dictionary of the model’s parameters.
# The model is saved in a folder called ./model. If this folder doesn’t exist, it’s created using os.makedirs.
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr       # Learning rate for training (controls how much the model weights change during training).
        self.gamma = gamma # Discount factor (used in Q-Learning to discount future rewards).
        self.model = model #model: The neural network model (instance of Linear_QNet).
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
    #    ?? #  Optimizer for updating the model parameters. Adam is an optimization algorithm that combines the benefits of SGD and RMSProp.
        self.criterion = nn.MSELoss()  # Loss function used to measure the difference between the predicted Q-values and the target Q-values. MSELoss is Mean Squared Error, used in regression problems like this one.

    def train_step(self, state, action, reward, next_state, done):
        # Converts state, next_state, action, and reward into PyTorch tensors of type float or long. This is required because PyTorch operates on tensors.
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

# Handling single experience:
# If the input state has only one dimension (meaning it's a single sample rather than a batch), it's reshaped to add a batch dimension using torch.unsqueeze. This ensures that the inputs are compatible with batch processing.

# handle multiple sizes
# you are passing a single state instead of a batch of states.
        if len(state.shape) == 1:
            # we want to append 
            # If state is a one-dimensional tensor (shape (x,)), the torch.unsqueeze() function is used to add an additional dimension to the tensor, converting it to shape (1, x). This makes it compatible for batch processing.
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state   represents the predicted value for each possible action (left, straight, or right).
        pred = self.model(state)

        target = pred.clone()  # A copy of the predicted Q-values is made using pred.clone() because we'll modify it.

        # This line starts a loop that will iterate over all experiences in the batch. Each experience corresponds to a tuple of (state, action, reward, next_state, done).
        for idx in range(len(done)):
            Q_new = reward[idx]

            # checks whether the game has ended for the current experience. If done[idx] is False, it means the episode is still ongoing.
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
    #  computes the maximum predicted Q-value for the next state, indicating the best possible action the agent can take next.
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            #  the entire line target[idx][torch.argmax(action[idx]).item()] = Q_new effectively does the following:

            # It updates the target Q-value for the action taken in the idx-th experience in the batch.
            # Specifically, it sets the Q-value for the action that the agent chose (the one corresponding to the maximum index in the one-hot encoded action) to be equal to the newly calculated Q_new value.

# This update is crucial for the Q-learning process because it allows the agent to learn from its experiences. By adjusting the target Q-value for the action taken to the new value Q_new, the model aims to reduce the difference between the predicted Q-values and the target Q-values during training. This helps the model learn which actions lead to better long-term rewards.
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()  # to empty the gradient
        loss = self.criterion(target, pred) # loss function q new , q   
        loss.backward() #computes the gradients of the loss with respect to the model parameters by backpropagation. This is where the model learns how to update its weights to minimize the loss.
        self.optimizer.step()
# computes the gradients of the loss with respect to the model parameters by backpropagation. This is where the model learns how to update its weights to minimize the loss.


# Prepares the input state for batch processing.
# Predicts Q-values for the current state.
# Updates target Q-values based on immediate rewards and expected future rewards.
# Computes the loss and backpropagates it to adjust the model weights for better performance in future actions.


