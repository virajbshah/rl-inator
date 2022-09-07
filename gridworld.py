from enum import Enum
from tkinter import Grid
import numpy as np

eps = 1e-9


class Action(Enum):
    N, S, E, W = (i for i in range(4))


class GridWorld:
    """ A test reinforcement learning environment where
        the agent is to learn to navigate around a grid
        with rewards determined by the tile it lands on
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.state = (0, 0)

        self.stateReward = np.zeros((rows, cols), np.float128)
        stateState = (np.random.randint(0, rows), np.random.randint(0, cols))
        negState = (np.random.randint(0, rows), np.random.randint(0, cols))
        while stateState == negState:
            negState = (np.random.randint(0, rows), np.random.randint(0, cols))
        self.stateReward[stateState] = 1
        self.stateReward[negState] = -1

    def afterAction(self, state, action):
        """ Returns the state that would be transitioned to
            after given action would be executed from the
            given statetion
        """

        if action == Action.N:
            return (max(0, state[0] - 1), state[1])
        elif action == Action.S:
            return (min(self.rows - 1, state[0] + 1), state[1])
        elif action == Action.E:
            return (state[0], min(self.cols - 1, state[1] + 1))
        elif action == Action.W:
            return (state[0], max(0, state[1] - 1))
        else:
            print('Invalid action.')
            return state

    def doAction(self, action):
        """ Changes state as per state transition kernel
            and returns numerical reward
        """

        self.state = self.afterAction(self.state, action)
        return self.stateReward[self.state]


rows, cols, gamma = 3, 3, 0.5
values = np.zeros((rows, cols), np.float128)
gw = GridWorld(rows, cols)

print(gw.stateReward)

while True:
    updatedValues = np.zeros((rows, cols), np.float128)

    it = np.nditer(values, flags=['multi_index'])
    for _ in it:
        possibleNextStates = [gw.afterAction(
            it.multi_index, action) for _, action in Action.__members__.items()]
        maxValue = max([values[possibleNextState]
                       for possibleNextState in possibleNextStates])
        nextStates = list(
            filter(lambda s: np.abs(values[s] - maxValue) <= eps, possibleNextStates))

        # print(f'{it.multi_index} -> {nextStates}')

        for nextState in nextStates:
            updatedValues[it.multi_index] += values[nextState]
        updatedValues[it.multi_index] *= gamma / len(nextStates)
        updatedValues[it.multi_index] += gw.stateReward[it.multi_index]

    if (np.abs(values - updatedValues) <= eps).all():
        break

    values = np.copy(updatedValues)

np.set_printoptions(2)
print(values)
