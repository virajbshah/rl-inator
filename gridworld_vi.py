from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

eps = 1e-9

class Action(Enum):
    N, S, E, W = (10 ** i for i in range(4))


class GridWorld:
    """ A test reinforcement learning environment where
        the agent is to learn to navigate around a grid
        with rewards determined by the tile it lands on
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.state = (0, 0)

        self.stateReward = np.zeros((rows, cols), np.float64)
        posState = (np.random.randint(0, rows), np.random.randint(0, cols))
        negState = (np.random.randint(0, rows), np.random.randint(0, cols))
        while posState == negState:
            negState = (np.random.randint(0, rows), np.random.randint(0, cols))
        self.stateReward[posState] = 1
        self.stateReward[negState] = -1

    def afterAction(self, state, action):
        """ Returns the state that would be transitioned to
            after given action would be executed from the
            given position
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
        """ Changes position as per state transition kernel
            and returns numerical reward
        """

        self.state = self.afterAction(self.state, action)
        return self.stateReward[self.state]


rows, cols, gamma = 5, 5, 0.5
values = np.zeros((rows, cols), np.float64)
gw = GridWorld(rows, cols)

print(gw.stateReward)

while True:
    updatedValues = np.zeros((rows, cols), np.float64)

    it = np.nditer(updatedValues, flags=['multi_index'], op_flags=['writeonly'])
    for updatedValue in it:
        possibleNextStates = [gw.afterAction(it.multi_index, action) for _, action in Action.__members__.items()]
        maxValue = max([values[possibleNextState] for possibleNextState in possibleNextStates])
        nextStates = list(filter(lambda s: np.abs(values[s] - maxValue) <= eps, possibleNextStates))

        for nextState in nextStates:
            updatedValue += values[nextState]
        updatedValue *= gamma / len(nextStates)
        updatedValue += gw.stateReward[it.multi_index]

    if (np.abs(values - updatedValues) <= eps).all():
        break

    values = np.copy(updatedValues)

np.set_printoptions(2)
print(values)

policy = np.zeros((rows, cols), '<U1')
it = np.nditer(values, flags=['multi_index'], op_flags=['writeonly'])
for _ in it:
    possibleNextStates = [gw.afterAction(it.multi_index, action) for _, action in Action.__members__.items()]
    maxValue = max([values[possibleNextState] for possibleNextState in possibleNextStates])

    actionCode = 0
    for _, action in Action.__members__.items():
        if np.abs(values[gw.afterAction(it.multi_index, action)] - maxValue) <= eps:
            actionCode += action._value_

    if actionCode == 1:
        policy[it.multi_index] = '↑'
    elif actionCode == 10:
        policy[it.multi_index] = '↓'
    elif actionCode == 100:
        policy[it.multi_index] = '→'
    elif actionCode == 1000:
        policy[it.multi_index] = '←'
    elif actionCode == 11:
        policy[it.multi_index] = '↕'
    elif actionCode == 1100:
        policy[it.multi_index] = '↔'
    elif actionCode == 1001:
        policy[it.multi_index] = '↖'     
    elif actionCode == 101:
        policy[it.multi_index] = '↗'     
    elif actionCode == 110:
        policy[it.multi_index] = '↘'     
    elif actionCode == 1010:
        policy[it.multi_index] = '↙'
    elif actionCode == 1111:
        policy[it.multi_index] = '✣'   
    else:
        policy[it.multi_index] = '?'

print(policy)

plt.imshow(values, cmap='coolwarm')
plt.show()