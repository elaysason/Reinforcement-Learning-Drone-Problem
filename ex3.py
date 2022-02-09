import random
import collections

all_actions = ['move_up', 'move_down', 'move_left', 'move_right','wait','reset']

class DroneAgent:
    def __init__(self, n, m, alpha=0.45, gamma=0.25,epsilon=0.15):
        self.mode = 'train' # do not change this!
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha  # learning constant
        self.gamma = gamma  # discount constant
        self.m = m
        self.n = n

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def actions(self, obs0):
        all_pos_actions = ['wait', 'reset']

        x, y = obs0['drone_location']

        if 0 <= x - 1 < self.m:
            all_pos_actions.append('move_up')
        if 0 <= x + 1 < self.m:
            all_pos_actions.append('move_down')
        if 0 <= y - 1 < self.m:
            all_pos_actions.append('move_left')
        if 0 <= y + 1 < self.m:
            all_pos_actions.append('move_right')

        for pack in obs0['packages']:
            if (x, y) == pack[1]:
                all_pos_actions.append('pick')
                continue

        if (x, y) == obs0['target_location']:
            for pack in obs0['packages']:
                if type(pack[1]) != tuple:
                    all_pos_actions.append('deliver')
                    continue
        return all_pos_actions
    def best_actions(self,obs0):
        possible_actions = self.actions(obs0)
        if len(obs0['packages']) == 0:
            return 'reset'

        else:
            possible_actions.remove('reset')
            state = repr(obs0)
            q = [self.get_q(state, a) for a in possible_actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(possible_actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = possible_actions[i]
            return action

    def select_action(self, obs0):
        possible_pos = self.actions(obs0)
        if self.mode == 'train':
            if random.uniform(0,1) < self.epsilon:
                return random.choice(possible_pos)
            else:
                return self.best_actions(obs0)
        else:
            return self.best_actions(obs0)

    def train(self):
        self.mode = 'train'  # do not change this!

    def eval(self):
        self.mode = 'eval'  # do not change this!

    def update(self, obs0, action, obs1, reward):
        curr_state = repr(obs0)
        next_state = repr(obs1)

        q_max = max([self.get_q(next_state, a) for a in self.actions(obs1)])
        old_q = self.q.get((curr_state, action), None)

        if old_q is None:
            self.q[(curr_state, action)] = reward
        else:
            self.q[(curr_state, action)] = old_q + self.alpha * (reward + self.gamma * q_max - old_q)
