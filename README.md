# DQN

This repo shows my implementation of a DQN agent that can tackle any maze with a set of unknown obstacles and goal locations. 

So, the approach here is to discretize actions and instead of looking at all possible directions we look at only 4 directions. The implementation of this can be found in `find action()` and `discrete_action_to_continuous()` methods. 

It is worth mentioning that choosing actions randomly was weighted, giving (up and down) slightly higher probability to occur than right, and assigning left a small weight **assuming** that the goal is always to the right of the agent. This has drastically improved the performance. 

With experiements, the following has been concluded:
1. Episode Size: The problem with large values was that the agent spend too much time in the goal area, and does not get the chance to explore the start of the maze more. So, a small (500) value was chosen as a fixed step size.
2. Epsilon Decay δ: small value (0.00009)
3. Target Network Update Frequency: (50) if the goal path is not found, (0)
otherwise (to stop training)
4. Batch Size: (256) shows a significant improvement over other values
5. Learning Rate α fixed to (0.005)
6. Discount Factor γ fixed to (0.99)

Yet, it is been noticed with experiments that epsilon will reach its minimum (0.05) from the very beginning of the training, and thus, the agent might follow a greedy but not a well-trained path. To avoid this problem, epsilon is resetted at the beginning of each new episode with a value equal to (previous episode epsilon - 0.15). This can be seen in the `has_finished_epsiode()` method in the agent class. 

Additionally, if the agent does not recognize the goal path (explained below) minimum epsilon (0.05) will be increased as the number of episodes increase to allow the agent to explore more. Furthermore, a new local variable has been introduced, `self.solved`, to check after each three episodes(`self.test_frequ`) if the greedy path is indeed the goal path; If that is the case, the target network will not be updated anymore, and the agent will always follow that greedy path.
