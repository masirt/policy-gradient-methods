import numpy as np
import torch.nn as nn
import torch.functional as F
import torch.optim as optimizer
import gym


class ParameterisedPolicyNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass

class ParameterisedValueFunctionNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass


def determine_action(S_t, parameterised_policy):
    pass

def generate_episode(env, parameterised_policy):
    S_t = env.reset()
    full_trajectory = []

    while True:

        A_t = determine_action(S_t, parameterised_policy)
        S_t_next, R_t, terminal, _ = env.step(A_t)

        full_trajectory += [(S_t, A_t, R_t)]

        if terminal:
            break

        S_t = S_t_next

    return full_trajectory

def update_w(delta, optimizer_w, parameterised_value_func, S_t):
    pass

def update_theta(delta, optimizer_theta, parameterised_policy, A_t, S_t):
    pass


def reinforce_with_baseline(alpha_w, alpha_theta, gamma, env, episodes):
    parameterised_policy = ParameterisedPolicyNetwork()
    parameterised_value_func = ParameterisedValueFunctionNetwork()

    optimizer_theta = optimizer.Adam(parameterised_policy.parameters(), lr=alpha_theta)
    optimizer_w = optimizer.Adam(parameterised_value_func.parameters(), lr=alpha_w)

    for episode in range(episodes):
        full_trajectory = generate_episode(env, parameterised_policy)
        trajectory_rewards = [step_vals[2] for step_vals in full_trajectory]
        for t, step_vals in enumerate(full_trajectory):
            S_t, A_t, R_t = step_vals
            G = np.sum([gamma**(k-t-1)*R_k for k, R_k in enumerate(trajectory_rewards[t+1:])])
            delta = G - parameterised_value_func(S_t)
            update_w(delta, optimizer_w, parameterised_value_func, S_t)
            update_theta(delta, optimizer_theta, parameterised_policy, A_t, S_T)
        print("EPISODE: {}, SUM OF REWARDS: {}".format(episode, np.sum(trajectory_rewards)))


def main():
    ALPHA_THETA = 0.01
    ALPHA_W = 0.01
    GAMMA = 0.99
    EPISODES = 200

    reinforce_with_baseline(alpha_w=ALPHA_W, alpha_theta=ALPHA_THETA, gamma=GAMMA, episodes=EPISODES)
    

if __name__=="__main__":
    main()