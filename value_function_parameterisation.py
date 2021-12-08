import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import torch
import numpy as np


class ParameterisedValueFunctionNetwork(nn.Module):

    def __init__(self, state_space_cardinality) -> None:
        super().__init__()
        self.input_fc = nn.Linear(in_features=state_space_cardinality, out_features=128)
        self.relu = nn.ReLU()
        # just one output as this is the parameterised value function
        self.output_fc = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.relu(x)
        x = self.output_fc(x)

        return x

class ParameterisedValueFunctionReinforce():

    def __init__(self, env,alpha_w, linear=True, M=5) -> None:
        self.env = env
        self.linear = linear
        self.state_space = self.env.observation_space.shape[0]
        self.number_of_actions = self.env.action_space.n
        
        self.alpha_w = alpha_w

        if self.linear:
            self.w = np.zeros((self.state_space*M,1))
            self.M = M
        else:
            self.value_func_network = ParameterisedValueFunctionNetwork(self.state_space)
            self.value_func_optimizer = optimizer.Adam(self.value_func_network.parameters(), lr=alpha_w)

    def phi(self, S_t):
        S_t = (S_t-self.env.observation_space.low)/(self.env.observation_space.high - self.env.observation_space.low)
        phi = np.zeros((self.state_space*self.M,1))
        i = 0
        for feature_index in range(self.state_space):
            for m in range(self.M):
                phi[i] = np.cos(m*np.pi*S_t[feature_index])
                i+=1
        return phi

    def compute_delta(self, G, S_t=None):
        if S_t is None:
            return G

        if self.linear:
            delta = G - self.w.T.dot(self.phi(S_t))
            return delta
        else:
            self.value_func_network.eval()
            delta = G -self.value_func_network(torch.tensor(S_t)).detach().numpy()
            return delta

    def update_w(self, delta, S_t):
        if self.linear:
            gradient = self.phi(S_t)
            self.w = self.w + self.alpha_w*delta*gradient
        else:
            self.value_func_network.train()
            self.value_func_optimizer.zero_grad()
            loss = torch.sum(- (torch.tensor(delta) * self.value_func_network(torch.tensor(S_t)))) # delta is just a constant that can be multiplied
            loss.backward()
            self.value_func_optimizer.step()
    

class ParameterisedValueFunctionActorCritic():

    def __init__(self, env,alpha_w, linear=True, M=3) -> None: 
        self.env = env
        self.linear = linear
        self.state_space = self.env.observation_space.shape[0]
        self.number_of_actions = self.env.action_space.n
        
        self.alpha_w = alpha_w

        if self.linear:
            self.w = np.zeros((self.state_space*M,1))
            self.M = M
        else:
            self.value_func_network = ParameterisedValueFunctionNetwork(self.state_space)
            self.value_func_optimizer = optimizer.Adam(self.value_func_network.parameters(), lr=alpha_w)

    def phi(self, S_t):
        S_t = (S_t-self.env.observation_space.low)/(self.env.observation_space.high - self.env.observation_space.low)
        phi = np.zeros((self.state_space*self.M,1))
        i = 0
        for feature_index in range(self.state_space):
            for m in range(self.M):
                phi[i] = np.cos(m*np.pi*S_t[feature_index])
                i+=1
        return phi

    def compute_delta(self, R_t, S_t, S_t_next, gamma, terminal):

        if self.linear:
            if not terminal:
                delta = R_t +  gamma * self.w.T.dot(self.phi(S_t_next)) - self.w.T.dot(self.phi(S_t))
            else:
                delta = R_t +  self.w.T.dot(self.phi(S_t)) 
            return delta
        else:
            self.value_func_network.eval()
            if not terminal:
                delta = R_t  + gamma * self.value_func_network(torch.tensor(S_t_next)).detach().numpy() - self.value_func_network(torch.tensor(S_t)).detach().numpy()
            else:
                delta = R_t + self.value_func_network(torch.tensor(S_t)).detach().numpy()
            return delta

    def update_w(self, delta, S_t):
        if self.linear:
            gradient = self.phi(S_t)
            self.w = self.w + self.alpha_w*delta*gradient
        else:
            self.value_func_network.train()
            self.value_func_optimizer.zero_grad()
            loss = torch.sum(- (torch.tensor(delta) * self.value_func_network(torch.tensor(S_t)))) # delta is just a constant that can be multiplied
            loss.backward()
            self.value_func_optimizer.step()
    
