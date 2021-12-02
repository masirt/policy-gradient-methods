import numpy as np
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
import torch.optim as optimizer
import torch
import gym


class ParameterisedPolicyNetwork(nn.Module):

    def __init__(self, state_space_cardinality, action_space_cardinality) -> None:
        super().__init__()
        self.input_fc = nn.Linear(in_features=state_space_cardinality, out_features=64)
        self.relu = nn.ReLU()
        self.output_fc = nn.Linear(
            in_features=64, out_features=action_space_cardinality)

    def forward(self, x):

        # pass state through the network
        x = self.input_fc(x)
        x = self.relu(x)
        x = F.softmax(self.output_fc(x))  # softmax added but do not know if it will work

        return x


class ParameterisedValueFunctionNetwork(nn.Module):

    def __init__(self, state_space_cardinality) -> None:
        super().__init__()
        self.input_fc = nn.Linear(in_features=state_space_cardinality, out_features=64)
        self.relu = nn.ReLU()
        # just one output as this is the parameterised value function
        self.output_fc = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.relu(x)
        x = self.output_fc(x)

        return x


def determine_action(S_t, parameterised_policy, action_space_cardinality):
    parameterised_policy.eval()
    pi_given_S = parameterised_policy(torch.tensor(S_t)).cpu().detach().numpy()
    # print("PI GIVEN S IN DET ACTION", pi_given_S)
    
    A_t = np.random.choice(action_space_cardinality, p=pi_given_S)
    return A_t


def generate_episode(env, parameterised_policy, action_space_cardinality):
    S_t = env.reset()
    full_trajectory = []

    while True:

        A_t = determine_action(S_t, parameterised_policy, action_space_cardinality)
        S_t_next, R_t, terminal, _ = env.step(A_t)

        full_trajectory += [(S_t, A_t, R_t)]

        if terminal:
            break

        S_t = S_t_next

    return full_trajectory


def update_w(delta, optimizer_w, parameterised_value_func, S_t):#, actor_critic = False, I = None):
    parameterised_value_func.train()
    optimizer_w.zero_grad()
    # if not actor_critic:
    loss_value = torch.sum(- (torch.tensor(delta) * parameterised_value_func(torch.tensor(S_t)))) # delta is just a constant that can be multiplied
    # else:
    #     loss_value = torch.sum(- (I*torch.tensor(delta) * parameterised_value_func(torch.tensor(S_t)))) # delta is just a constant that can be multiplied
    loss_value.backward()
    optimizer_w.step()


def update_theta(delta, optimizer_theta, parameterised_policy, A_t, S_t, gamma, t=None, actor_critic = False, I=None):
    parameterised_policy.train()
    optimizer_theta.zero_grad()
    pi_given_S = parameterised_policy(torch.tensor(S_t))
    # print("PI GIVEN S UPDATE", pi_given_S[A_t])
    if not actor_critic:
        loss_policy = torch.sum(- (gamma**t * torch.tensor(delta) * torch.log(pi_given_S[A_t])))
    else:
        loss_policy = torch.sum(- (I* torch.tensor(delta)* torch.log(pi_given_S[A_t])))
    loss_policy.backward()
    optimizer_theta.step()

def reinforce_with_baseline(alpha_w, alpha_theta, gamma, env, episodes):
    state_space_cardinality = env.observation_space.shape[0]
    action_space_cardinality = env.action_space.n
    parameterised_policy = ParameterisedPolicyNetwork(state_space_cardinality=state_space_cardinality, action_space_cardinality=action_space_cardinality)
    parameterised_value_func = ParameterisedValueFunctionNetwork(state_space_cardinality=state_space_cardinality)

    optimizer_theta = optimizer.Adam(
        parameterised_policy.parameters(), lr=alpha_theta)
    optimizer_w = optimizer.Adam(
        parameterised_value_func.parameters(), lr=alpha_w)
    accumulated_rewards = []

    for episode in range(episodes):
        full_trajectory = generate_episode(env, parameterised_policy, action_space_cardinality)
        trajectory_rewards = [step_vals[2] for step_vals in full_trajectory]
        for t, step_vals in enumerate(full_trajectory):
            S_t, A_t, R_t = step_vals
            G = np.sum([gamma**(k-t-1)*R_k for k,
                       R_k in enumerate(trajectory_rewards[t+1:])])
            parameterised_value_func.eval()
            delta = G - parameterised_value_func(torch.tensor(S_t)).detach().numpy()

            # print("DELTA VAL", delta_value, "DELTA POL", delta_policy)
            update_w(delta, optimizer_w, parameterised_value_func, S_t)
            update_theta(delta, optimizer_theta, parameterised_policy, A_t, S_t, gamma, t)
        if len(accumulated_rewards) >= 100:
            accumulated_rewards[episode%len(accumulated_rewards)] = np.sum(trajectory_rewards)
        else:
            accumulated_rewards.append(np.sum(trajectory_rewards))

        print("EPISODE: {}, SUM OF REWARDS: {}, ACC SUM REWS {}".format(
            episode, np.sum(trajectory_rewards), np.mean(accumulated_rewards)))
        # print("\n PARAMS VAL FUNC",[par.data for _, par in parameterised_value_func.named_parameters()], "\n")
        # print("\n PARAMS Policy",parameterised_policy.named_parameters(), "\n")


def one_step_actor_critic(alpha_w, alpha_theta, gamma, env, episodes):
    state_space_cardinality = env.observation_space.shape[0]
    action_space_cardinality = env.action_space.n
    parameterised_policy = ParameterisedPolicyNetwork(state_space_cardinality=state_space_cardinality, action_space_cardinality=action_space_cardinality)
    parameterised_value_func = ParameterisedValueFunctionNetwork(state_space_cardinality=state_space_cardinality)

    optimizer_theta = optimizer.Adam(
        parameterised_policy.parameters(), lr=alpha_theta)
    optimizer_w = optimizer.Adam(
        parameterised_value_func.parameters(), lr=alpha_w)
    accumulated_rewards = []
    
    for episode in range(episodes):
        S_t = env.reset()
        I = 1
        accumulated_reward = 0
        while True:
            A_t = determine_action(S_t, parameterised_policy, action_space_cardinality)
            S_t_next, R_t, terminal, _ = env.step(A_t)
            accumulated_reward += R_t

            parameterised_value_func.eval()
            if not terminal:
                delta = R_t + gamma*parameterised_value_func(torch.tensor(S_t_next)).detach().numpy() - parameterised_value_func(torch.tensor(S_t)).detach().numpy() 
            else:
                delta = R_t + parameterised_value_func(torch.tensor(S_t)).detach().numpy()
            update_w(delta, optimizer_w, parameterised_value_func, S_t)#, actor_critic= True, I=I)
            update_theta(delta, optimizer_theta, parameterised_policy, A_t, S_t, gamma, actor_critic = True, I=I)

            

            if terminal:
                break
            I *= gamma
            S_t = S_t_next

            

        if len(accumulated_rewards) >= 100:
            accumulated_rewards[episode%len(accumulated_rewards)] = accumulated_reward
        else:
            accumulated_rewards.append(accumulated_reward)

        print("EPISODE: {}, SUM OF REWARDS: {}, ACC SUM REWS {}".format(
            episode, accumulated_reward, np.mean(accumulated_rewards)))


def main():
    print(2e-14)
    ALPHA_THETA = 2**-12
    ALPHA_W = 2**-9

    # ALPHA_theta, ALPHA_w for REINFORCE w b: 12,9 works; 10,7 works; 9,7 works kind off but very unstable; 10,8 works 
    GAMMA = 0.99
    EPISODES = 2000
    print(ALPHA_W, ALPHA_THETA)

    env = gym.make('CartPole-v0')

    reinforce_with_baseline(
        alpha_w=ALPHA_W, alpha_theta=ALPHA_THETA, gamma=GAMMA,env = env, episodes=EPISODES)
    # one_step_actor_critic(
    #     alpha_w=ALPHA_W, alpha_theta=ALPHA_THETA, gamma=GAMMA,env = env, episodes=EPISODES)


if __name__ == "__main__":
    main()
