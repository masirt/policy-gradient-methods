import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import torch
import numpy as np
from scipy.special import softmax

class Parameterisedpolicy_network(nn.Module):

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

class ParameterisedPolicyReinforce():

    def __init__(self, env, alpha_theta, linear=True, M=1) -> None:
        self.env = env
        self.linear = linear
        self.state_space = self.env.observation_space.shape[0]
        self.number_of_actions = self.env.action_space.n
        self.alpha_theta = alpha_theta

        if self.linear:
            self.theta = np.zeros((self.state_space*self.number_of_actions*M, 1))
            self.M = M
        else:
            
            self.policy_network = Parameterisedpolicy_network(self.state_space, self.number_of_actions)
            self.policy_optimizer = optimizer.Adam(self.policy_network.parameters(), lr = self.alpha_theta)

    def phi_sigma_a(self, S_t, A_t):
        #S_t=(S_t-self.env.observation_space.low)/(self.env.observation_space.high - self.env.observation_space.low)
        phi_sigma_a = np.zeros((self.number_of_actions, self.state_space*self.M))
        phi_sigma_a[A_t] = S_t
        # phi_sigma_a = np.zeros((self.number_of_actions,self.state_space*self.M ))
        
        # i = 0
        # for feature_index in range(self.state_space):
        #     for m in range(self.M):
        #         phi_sigma_a[A_t,i] = np.cos(m*np.pi*S_t[feature_index])
        #         i+=1

        return phi_sigma_a.reshape((-1,1))


    def pi_given_S(self, S_t):
        
        pi_given_S = [np.dot(self.theta.T, self.phi_sigma_a(S_t, a)) for a in range(self.number_of_actions)]
        return softmax(pi_given_S)

    def pi_given_S_A(self, S_t, A_t):
        return softmax(np.dot(self.theta.T, self.phi_sigma_a(S_t,A_t)))

    def determine_action(self,S_t):
        if self.linear:
            pi_given_S = [wrapped[0][0] for wrapped in self.pi_given_S(S_t)]
            #print("PI G S", pi_given_S)
        else:
            self.policy_network.eval()
            pi_given_S = self.policy_network(torch.tensor(S_t)).cpu().detach().numpy()
        A_t = np.random.choice(self.number_of_actions, p=pi_given_S)
        return A_t
    
    def update_theta(self, delta, A_t, S_t, gamma, t):
        if self.linear:
            policy_times_phi = [self.pi_given_S_A(S_t, a)*self.phi_sigma_a(S_t,a) for a in range(self.number_of_actions)] # does this always work? TODO: Actions may not be numbering

            gradient = self.phi_sigma_a(S_t, A_t) - np.sum(policy_times_phi)
            self.theta = self.theta + self.alpha_theta*gamma**t*delta*gradient
        else:
            self.policy_network.train()
            self.policy_optimizer.zero_grad()
            pi_given_S = self.policy_network(torch.tensor(S_t))
            loss = torch.sum(- (gamma**t * torch.tensor(delta) * torch.log(pi_given_S[A_t])))
            loss.backward()
            self.policy_optimizer.step()

class ParameterisedPolicyActorCritic():

    def __init__(self, env, alpha_theta, linear=True, M=1) -> None:
        self.env = env
        self.linear = linear
        self.state_space = self.env.observation_space.shape[0]
        self.number_of_actions = self.env.action_space.n
        self.alpha_theta = alpha_theta

        if self.linear:
            self.theta = np.zeros((self.state_space*self.number_of_actions*M, 1))
            self.M = M
        else:
            
            self.policy_network = Parameterisedpolicy_network(self.state_space, self.number_of_actions)
            self.policy_optimizer = optimizer.Adam(self.policy_network.parameters(), lr = self.alpha_theta)

    def phi_sigma_a(self, S_t, A_t):
        #S_t=(S_t-self.env.observation_space.low)/(self.env.observation_space.high - self.env.observation_space.low)
        phi_sigma_a = np.zeros((self.number_of_actions, self.state_space*self.M))
        phi_sigma_a[A_t] = S_t
        # phi_sigma_a = np.zeros((self.number_of_actions,self.state_space*self.M ))
        
        # i = 0
        # for feature_index in range(self.state_space):
        #     for m in range(self.M):
        #         phi_sigma_a[A_t,i] = np.cos(m*np.pi*S_t[feature_index])
        #         i+=1

        return phi_sigma_a.reshape((-1,1))


    def pi_given_S(self, S_t):
        
        pi_given_S = [np.dot(self.theta.T, self.phi_sigma_a(S_t, a)) for a in range(self.number_of_actions)]
        return softmax(pi_given_S)

    def pi_given_S_A(self, S_t, A_t):
        return softmax(np.dot(self.theta.T, self.phi_sigma_a(S_t,A_t)))

    def determine_action(self,S_t):
        if self.linear:
            pi_given_S = [wrapped[0][0] for wrapped in self.pi_given_S(S_t)]
            #print("PI G S", pi_given_S)
        else:
            self.policy_network.eval()
            pi_given_S = self.policy_network(torch.tensor(S_t)).cpu().detach().numpy()
        A_t = np.random.choice(self.number_of_actions, p=pi_given_S)
        return A_t
    
    def update_theta(self, delta, A_t, S_t, I):
        if self.linear:
            policy_times_phi = [self.pi_given_S_A(S_t, a)*self.phi_sigma_a(S_t,a) for a in range(self.number_of_actions)] # does this always work? TODO: Actions may not be numbering

            gradient = self.phi_sigma_a(S_t, A_t) - np.sum(policy_times_phi)
            self.theta = self.theta + self.alpha_theta*I*delta*gradient
        else:
            self.policy_network.train()
            self.policy_optimizer.zero_grad()
            pi_given_S = self.policy_network(torch.tensor(S_t))
            loss = torch.sum(- (I* torch.tensor(delta) * torch.log(pi_given_S[A_t])))
            loss.backward()
            self.policy_optimizer.step()
