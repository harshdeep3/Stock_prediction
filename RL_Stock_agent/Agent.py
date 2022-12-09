import numpy as np


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # discount rate
        self.gamma = 0.95
        # exploration rate
        self.epsilon = 0.9
        self.epsilon_min = 0.01  # -> smallest epsilon value
        # after a testing the epsilon value goes 0.9 to around 0.01, exactly
        # what i want. So the decay is set to 0.998
        self.epsilon_decay = 0.9975  # -> decay epslion by 99.9%
        self.model = lm.LinearModel(state_size, action_size)

    def act(self, state):
        """[summary]
        This gets the actions which will be chosen. This can be a random action
        (explore) or the action which gets the best rewards (exploit).
        Args:
                state ([type]):The state describe the state stocks owned, OHLC prices,
                                            indicators, and cash_left
        Returns:
                [type]: The choosen action
        """
        if np.random.rand() <= self.epsilon:
            # exploring
            return np.random.choice(self.action_size)
        # exploiting
        act_values = self.model.predict(state).detach().numpy()
        return np.argmax(act_values)

    def train(self, state, action, reward, next_state, done):
        """[summary]
        This trains the model, gets the target value. The epsilon value decreased each time this is run
        Args:
                state ([vector]): The current state describe the environment using
                                                    the state vector
                action ([type]): The action that will be taken.
                reward ([type]): The change between the current protfolio and the
                                                one after the aciton is taken
                next_state ([vector]): Observation of the environment after the action
                                                        is taken
                done (boolean): If the data is on the last day
        """
        # get the prediction of the next state
        # this yhat = r + gamma * max (preds according to the actions)
        target = reward + self.gamma * \
                 np.amax(self.model.predict(next_state).detach().numpy())
        # get the predict of the y
        target_full = self.model.predict(state).detach().numpy()
        # q(s,a) = yhat
        # q(s,a) is [[x, y, z]], the 0 is to get the vector inside
        # the action gets the q(s,a) and updates it with the target (yhat)
        target_full[0, action] = target
        # calls SGD
        self.model.sgd(state, target_full)
        # decay of the explore rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """[summary]
        Load the previously saved weight, saving the need to retrain the agent
        Args:
                name ([type]): [description] name of the file which store the weights
        """
        self.model.load_weights(name)

    def save(self, name):
        """[summary]
        This save the weights so they can be use later.
        Args:
                name ([type]): [description] filename where the weight will be saved.
        """
        self.model.save_weights(name)