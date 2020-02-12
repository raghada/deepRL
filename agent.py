############################################################################
#################### RAGHAD ALGHONAIM - CID: 01610136 ######################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################
import numpy as np
import torch
from collections import deque

class Agent:
    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.episode_length = 500
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        #two networks, a target and policy one. 
        self.q_policy_network = Network(input_dimension=2, output_dimension=4)
        self.q_target_network = Network(input_dimension=2, output_dimension=4)

        # the frequencuy of changing target network
        self.update_frequency = 50

        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.gamma = 0.99
        #epsilon start value
        self.epsilon = 1
        #epsilon decay
        self.delta = 0.00009
        #minimum possible value of epsilon during training
        self.min_epsilon = 0.05
        #to reset epsilon at the beginning of each episode (prev-epsilon_decay)
        self.previous_epsilon = 1.15
        self.epsilon_decay = 0.15

        #to check if the solution has been found, true if found, false otherwise
        self.solved = False
        #frequency of checking
        self.test_frequ = 1500
        #to check if the steps to goal are less than 100, otherwise it is not a good path
        self.steps_to_goal = 0

        #buffer parameters
        self.batch_size = 256
        self.memory = ReplayBuffer()

        #optimizer
        self.learning_rate = 0.005
        self.optimiser = torch.optim.Adam(self.q_policy_network.parameters(), lr=self.learning_rate)
        
        #to store the current state distance to goal
        self.distance_to_goal = float('inf')

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        #check if the episode has finished
        if self.num_steps_taken % self.episode_length == 0:
            #reset epsilon for the new episode
            self.epsilon = max(self.min_epsilon, round(self.previous_epsilon - self.epsilon_decay,2))
            self.previous_epsilon = self.epsilon
            #if this is a test episode
            if self.num_steps_taken % self.test_frequ == 0:
                self.solved = True
                #temproraliy stop training
                self.update_frequency = 0
            if self.num_steps_taken % (10*self.episode_length) == 0 and not self.solved:
                self.min_epsilon = min(self.min_epsilon+0.05, 1)
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        """return the next action for the current state
        
        Arguments:
            state {Numpy Array} -- current state
        
        Returns:
            Numpy Array -- next action
        """
        action = self.find_action(state)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return self._discrete_action_to_continuous(action)

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        """set next state and continue training
        
        Arguments:
            next_state {Numpy Array} -- next state
            distance_to_goal {float} -- distance between next_state and goal_state
        """
        #check if the goal has been reached in less than 100 steps, stop training, otherwise, restart it
        if self.solved:
            self.steps_to_goal+=1
            if self.steps_to_goal<=100 and distance_to_goal < 0.03:
                self.epsilon = 0
                self.min_epsilon = 0
                self.epsilon_decay = 0
                self.previous_epsilon = 0
                self.min_epsilon = 0
                self.update_frequency = 0
            elif self.steps_to_goal > 100 and self.epsilon!=0:
                self.solved = False
                self.steps_to_goal = 0
                self.update_frequency = 50

        self.distance_to_goal = distance_to_goal
        # Convert the distance to a reward
        reward = (1 - (4*distance_to_goal))**3
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        #append transition to buffer and train 
        self.memory.append(transition)
        if len(self.memory) >= self.batch_size:
            batch = self.memory.sample(self.batch_size)
            _ = self.train_network(batch)
            #update epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon-self.delta)
            #update target network
            if self.update_frequency !=0:
                if self.num_steps_taken % self.update_frequency == 0:
                    self.update_target_network()


    def train_network(self, batch):
        """trains the policy net with new batch
        
        Arguments:
            batch {NumpyArray} -- batch of transitions
        
        Returns:
            float -- loss
        """
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(batch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

 
    def find_action(self, state):
        """set the next action
        
        Arguments:
            state {Numpy Array} -- current state
        
        Returns:
            [NumpyArray] -- discrete action [0,1,2,3] for [right, up, left, down]
        """
        #if goal path has found, return the greedy action
        #otherwise, use epsilon greedy to decide on the action
        if self.solved:
            return [self.q_target_network.forward(torch.tensor(state)).argmax()]
        elif np.random.random() >= self.epsilon:
            q = self.q_policy_network.forward(torch.tensor(state)).argmax()
            return [q.item()]
        else:
            values = [0,1,2,3]
            return np.random.choice(values, 1, p=[0.3, 0.325, 0.05, 0.325])

 
    def _discrete_action_to_continuous(self, discrete_action):
        """convert discrete action to continus
        
        Arguments:
            discrete_action {NumpyArray} -- discrete action
        
        Returns:
            NumpyArray -- continuos action
        """
        continuous_actions = np.array([[0.02, 0], [0, 0.02], [-0.02,0], [0, -0.02]], dtype=np.float32)
        return continuous_actions[discrete_action[0]]
  

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        """returns the greedy action of a state
        
        Arguments:
            state {NumpyArray} -- current state
        
        Returns:
            NumpyArray -- greedy action of state
        """
        q = self.q_target_network.forward(torch.tensor(state)).argmax()
        action = self._discrete_action_to_continuous([q.item()])
        return action


    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        """computes loss
        
        Arguments:
            transition {tuple} -- a transition tuple to be used to compute loss
        
        Returns:
            MSELoss
        """
        states, actions, rewards, next_states = transition
        q_next_state = self.q_target_network.forward(next_states).detach().max(1)[0].unsqueeze(1)
        q_real = rewards + self.gamma*q_next_state
        network_prediction = self.q_policy_network.forward(states)
        q_exptected = torch.gather(network_prediction, 1, actions)
        loss = torch.nn.MSELoss()(q_exptected, q_real)
        return loss

 
    def update_target_network(self):
        """update target network 
        """
        self.q_target_network.load_state_dict(self.q_policy_network.state_dict())

class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=300)
        self.layer_2 = torch.nn.Linear(in_features=300, out_features=300)
        self.layer_3 = torch.nn.Linear(in_features=300, out_features=150)
        self.layer_4 = torch.nn.Linear(in_features=150, out_features=150)
        self.output_layer = torch.nn.Linear(in_features=150, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))
        output = self.output_layer(layer_4_output)
        return output

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=10**6)
    
    def append(self, transition):
        self.buffer.append(transition)

    
    def sample(self, batch_size):
        batch_indeces = np.random.choice(range((len(self.buffer))), batch_size, replace=False)
        batch = []
        for i in batch_indeces:
            batch.append(self.buffer[i])

        states = torch.from_numpy(np.vstack([t[0] for t in batch if t is not None])).float()
        actions = torch.from_numpy(np.vstack([t[1] for t in batch if t is not None])).long()
        rewards = torch.from_numpy(np.vstack([t[2] for t in batch if t is not None])).float()
        next_states = torch.from_numpy(np.vstack([t[3] for t in batch if t is not None])).float()
        
        return (states, actions, rewards, next_states)

    def __len__(self):
        return len(self.buffer)