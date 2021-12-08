import numpy as np
import random


class ActionSpace:
    def __init__(self) -> None:
        self.n = 4

class GridWorld687:

    def __init__(self) -> None:
        self.upper_limit = 4
        self.lower_limit = 0
        self.terminal_state = [4, 4]
        self.water_state = [4, 2]
        self.obstacle1 = [2, 2]
        self.obstacle2 = [3, 2]
        self.observation_space = np.zeros((25,))
        self.action_space = ActionSpace()
        

    def step(self, a):
        curr_pos = self.curr_pos.copy()

        new_pos = curr_pos.copy()
        if a == 0:  # up
            if random.uniform(0, 1) <= 0.05:
                new_pos = [curr_pos[0], curr_pos[1]+1]
            elif random.uniform(0, 1) <= 0.05:
                new_pos = [curr_pos[0], curr_pos[1] - 1]

            elif random.uniform(0, 1) <= 0.1:
                new_pos = curr_pos
            else:
                new_pos = [curr_pos[0]-1, curr_pos[1]]
        elif a == 1:  # left
            if random.uniform(0, 1) <= 0.05:
                new_pos = [curr_pos[0]-1, curr_pos[1]]
            elif random.uniform(0, 1) <= 0.05:
                new_pos = [curr_pos[0]+1, curr_pos[1]]
            elif random.uniform(0, 1) <= 0.1:
                new_pos = curr_pos
            else:
                new_pos = [curr_pos[0], curr_pos[1]-1]
        elif a == 2:  # down
            if random.uniform(0, 1) <= 0.05:
                new_pos = [curr_pos[0], curr_pos[1]-1]
            elif random.uniform(0, 1) <= 0.05:
                new_pos = [curr_pos[0], curr_pos[1] + 1]

            elif random.uniform(0, 1) <= 0.1:
                new_pos = curr_pos
            else:
                new_pos = [curr_pos[0]+1, curr_pos[1]]
        elif a == 3:  # right

            if random.uniform(0, 1) <= 0.05:
                new_pos = [curr_pos[0]+1, curr_pos[1]]
            elif random.uniform(0, 1) <= 0.05:
                new_pos = [curr_pos[0]-1, curr_pos[1]]
            elif random.uniform(0, 1) <= 0.1:
                new_pos = curr_pos
            else:
                new_pos = [curr_pos[0], curr_pos[1]+1]

        # Edge case handling of new pos
        if new_pos[0] > self.upper_limit or new_pos[1] > self.upper_limit:
            new_pos = curr_pos
        if new_pos[0] < self.lower_limit or new_pos[1] < self.lower_limit:
            new_pos = curr_pos

        if new_pos == self.obstacle1 or new_pos == self.obstacle2:
            new_pos = curr_pos

        # reward earned by getting new pos
        reward = 0
        done = False
        if new_pos == self.water_state:
            reward = -10
        elif curr_pos != self.terminal_state and new_pos == self.terminal_state:
            reward = 10
            done = True
        
        if curr_pos == self.terminal_state:
            done = True
            reward = 0
            new_pos = self.terminal_state

        # set done True if reached terminal state
        self.curr_pos = new_pos
        return new_pos, reward, done, None

    def reset(self):
        self.curr_pos = [0, 0]
        self.curr_pos[1] = np.random.choice([0, 1, 2, 3, 4])
        if self.curr_pos[1] == 2:
            self.curr_pos[0] = np.random.choice([0, 1, 4])
        else:
            self.curr_pos[0] = np.random.choice([0, 1, 2, 3, 4])

        return self.curr_pos