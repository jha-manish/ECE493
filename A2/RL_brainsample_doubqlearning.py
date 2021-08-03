import numpy as np
import pandas as pd


class rlalgorithm:
    def __init__(self, actions, lr = 0.01 , gamma = 0.9, epsilon = 0.1):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_a = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_b = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name = 'DoubleQLearning'

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_a = self.q_table_a.loc[observation, :]
            state_b = self.q_table_b.loc[observation, :]
            state_sum = state_a + state_b
            action = np.random.choice(state_sum[state_sum == np.max(state_sum)].index)

        return action


    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        a_ = self.choose_action(str(s_))

        if np.random.uniform() < 0.5:
            action = self.q_table_a.loc[s_, :]
            action_idx = np.random.choice(action[action == np.max(action)].index)
            self.q_table_a.loc[s, a] += self.lr * (r + self.gamma * self.q_table_b.loc[s_, action_idx] - self.q_table_a.loc[s, a])
        else:
            action = self.q_table_b.loc[s_, :]
            action_idx = np.random.choice(action[action == np.max(action)].index)
            self.q_table_b.loc[s, a] += self.lr * (r + self.gamma * self.q_table_a.loc[s_, action_idx] - self.q_table_b.loc[s, a])

        return s_, a_

    def check_state_exist(self, state):
        if state not in self.q_table_a.index:
            self.q_table_a = self.q_table_a.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table_a.columns,
                    name=state,
                )
            )

        if state not in self.q_table_b.index:
            self.q_table_b = self.q_table_b.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table_b.columns,
                    name=state,
                )
            )
