from actor_critic import OneStepActorCritic
from policy_parameterisation import ParameterisedPolicyActorCritic, ParameterisedPolicyReinforce
from value_function_parameterisation import ParameterisedValueFunctionActorCritic, ParameterisedValueFunctionReinforce
from reinforce import Reinforce

import matplotlib.pyplot as plt
import gym
import numpy as np
from scipy.signal import lfilter



def reinforce_baseline_hyperparameter_plot_cartpole_nn():
    alpha_configs = [(2**-8, 2**-6),(2**-10, 2**-7),(2**-12, 2**-9), (2**-14, 2**-11), (2**-16,2**-13), (2**-18, 2**-15)]
    env = gym.make("CartPole-v0")
    cdict = plt.cm.get_cmap("Set2")
    
    for i, alphas in enumerate(alpha_configs):
        alpha_theta, alpha_w = alphas

        parameterised_value_func = ParameterisedValueFunctionReinforce(env=env, alpha_w=alpha_w, linear=False)
        parameterised_policy = ParameterisedPolicyReinforce(env, alpha_theta, False)
        reinforce_agent = Reinforce(env, parameterised_policy, parameterised_value_func, baseline=True)

        accumulated_rewards = reinforce_agent.train(episodes = 500, gamma=.99)
        
        label = r"$\alpha^w$,$\alpha^\theta$ = $2^{"+str(np.log2(alpha_w).astype(np.int8))+r"}, 2^{"+str(np.log2(alpha_theta).astype(np.int8))+r"}$"
        n = 10  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        smoothed_acc_rewards = lfilter(b,a,accumulated_rewards)
        
        
        plt.plot( np.arange(len(accumulated_rewards)),accumulated_rewards, label=label, c=cdict(i), alpha=.5)
        plt.plot( np.arange(len(accumulated_rewards)),smoothed_acc_rewards, c=cdict(i))#,label=label)

    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/cartpole/hyperparameter_plot_cartpole_nn.png")


def reinforce_baseline_or_not_cartpole_nn():
    alpha_theta = 2**-12
    alpha_w= 2**-9

    parameterised_value_func_baseline = ParameterisedValueFunctionReinforce(env=gym.make("CartPole-v0"), alpha_w=alpha_w, linear=False)
    parameterised_policy_baseline = ParameterisedPolicyReinforce(env=gym.make("CartPole-v0"), alpha_theta= alpha_theta, linear=False)
    reinforce_agent_baseline = Reinforce(gym.make("CartPole-v0"), parameterised_policy_baseline, parameterised_value_func_baseline, baseline=True)

    accumulated_rewards_baseline = reinforce_agent_baseline.train(500, 0.99)

    parameterised_value_func_no_baseline = ParameterisedValueFunctionReinforce(env=gym.make("CartPole-v0"), alpha_w=alpha_w, linear=False)
    parameterised_policy_no_baseline = ParameterisedPolicyReinforce(env=gym.make("CartPole-v0"), alpha_theta= alpha_theta, linear=False)
    reinforce_agent_no_baseline = Reinforce(gym.make("CartPole-v0"), parameterised_policy_no_baseline, parameterised_value_func_no_baseline, baseline=False)


    accumulated_rewards_no_baseline = reinforce_agent_no_baseline.train(500, .99)

    
    cdict = plt.cm.get_cmap("Set2")

    n = 10  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    smoothed_acc_rewards = lfilter(b,a,accumulated_rewards_baseline)
    smoothed_acc_rewards_no_baseline = lfilter(b,a, accumulated_rewards_no_baseline)
    
    plt.plot( np.arange(len(accumulated_rewards_baseline)),accumulated_rewards_baseline, label="with baseline", c=cdict(2), alpha=.5)
    plt.plot( np.arange(len(accumulated_rewards_baseline)),smoothed_acc_rewards, c=cdict(2))#,label=label)

    plt.plot( np.arange(len(accumulated_rewards_no_baseline)),accumulated_rewards_no_baseline, label="without baseline", c=cdict(0), alpha=.5)
    plt.plot( np.arange(len(accumulated_rewards_no_baseline)),smoothed_acc_rewards_no_baseline, c=cdict(0))#,label=label)

    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/cartpole/reinforce_cartpole_baseline_no.png")

def basic_reinforce_linear():
    alpha_theta = 0.01
    alpha_w= 0.01
    env = gym.make("CartPole-v0")
    #env.observation_space.

    parameterised_policy_baseline = ParameterisedPolicyReinforce(env, alpha_theta= alpha_theta, linear=True)
    parameterised_value_func_baseline = ParameterisedValueFunctionReinforce(env, alpha_w=alpha_w, linear=True )

    reinforce_agent_baseline = Reinforce(env, parameterised_policy_baseline, parameterised_value_func_baseline, baseline=True)

    accumulated_rewards = reinforce_agent_baseline.train(500, 0.99)
    cdict = plt.cm.get_cmap("Set2")


    # plt.plot( np.arange(len(accumulated_rewards)),accumulated_rewards, label="Linear REINFORCE with baseline", c=cdict(0))
    # plt.xlabel("Episodes")
    # plt.ylabel("Accumulated Reward per episode")
 
    # plt.legend()
    # plt.savefig("figures/cartpole/reinforce_baseline_cartpole_linear.png")

def basic_actor_critic_linear():
    alpha_theta = 0.00000001
    alpha_w = 0.01 #0.01
    env = gym.make("CartPole-v0")
    cdict = plt.cm.get_cmap("Set2")


    parameterised_policy = ParameterisedPolicyActorCritic(env, alpha_theta, linear=True)
    parameterised_v = ParameterisedValueFunctionActorCritic(env, alpha_w, linear=True)

    actor_critic_agent = OneStepActorCritic(env, parameterised_policy, parameterised_v)
    accumulated_rewards = actor_critic_agent.train(500)

    plt.plot( np.arange(len(accumulated_rewards)),accumulated_rewards, label="Linear one step actor critic with baseline", c=cdict(0))
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/cartpole/actorcritic_cartpole_linear.png")



def main():
    #reinforce_baseline_hyperparameter_plot_cartpole_nn()
    #reinforce_baseline_or_not_cartpole_nn()
    np.random.seed(0)
    #basic_reinforce_linear()
    basic_actor_critic_linear()
if __name__=="__main__":
    main()