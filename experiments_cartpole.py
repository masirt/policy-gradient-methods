from actor_critic import OneStepActorCritic
from policy_parameterisation import ParameterisedPolicyActorCritic, ParameterisedPolicyReinforce
from value_function_parameterisation import ParameterisedValueFunctionActorCritic, ParameterisedValueFunctionReinforce
from reinforce import Reinforce

import matplotlib.pyplot as plt
import gym
import numpy as np
from scipy.signal import lfilter
import pandas as pd
import torch

#######         REINFORCE EXPERIMENTS           ######
def reinforce_baseline_hyperparameter_plot_nn_cartpole():
    alpha_configs = [(2**-8, 2**-6),(2**-10, 2**-7),(2**-12, 2**-9), (2**-14, 2**-11), (2**-16,2**-13), (2**-18, 2**-15)]
    env = gym.make("CartPole-v0")
    cdict = plt.cm.get_cmap("Set2")

    accumulated_rewards_all = []
    
    for i, alphas in enumerate(alpha_configs):
        alpha_theta, alpha_w = alphas

        parameterised_value_func = ParameterisedValueFunctionReinforce(env=env, alpha_w=alpha_w, linear=False)
        parameterised_policy = ParameterisedPolicyReinforce(env, alpha_theta, False)
        reinforce_agent = Reinforce(env, parameterised_policy, parameterised_value_func, baseline=True)

        accumulated_rewards, _ = reinforce_agent.train(episodes = 300, gamma=.99)
        
        label = r"$\alpha^w$,$\alpha^\theta$ = $2^{"+str(np.log2(alpha_w).astype(np.int8))+r"}, 2^{"+str(np.log2(alpha_theta).astype(np.int8))+r"}$"
        n = 10  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        smoothed_acc_rewards = lfilter(b,a,accumulated_rewards)
        
        
        plt.plot( np.arange(len(accumulated_rewards)),accumulated_rewards, label=label, c=cdict(i), alpha=.5)
        plt.plot( np.arange(len(accumulated_rewards)),smoothed_acc_rewards, c=cdict(i))#,label=label)

        accumulated_rewards_all.append(accumulated_rewards)


    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/cartpole/hyperparameter_plot_cartpole_nn.png")

    plt.clf()
    
    for i,accumulated_rewards in enumerate(accumulated_rewards_all):
        alpha_w, alpha_theta = alpha_configs[i]
        label = r"$\alpha^w$,$\alpha^\theta$ = $2^{"+str(np.log2(alpha_w).astype(np.int8))+r"}, 2^{"+str(np.log2(alpha_theta).astype(np.int8))+r"}$"
        n = 10  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        smoothed_acc_rewards = lfilter(b,a,accumulated_rewards)
        plt.plot( np.arange(len(accumulated_rewards)),smoothed_acc_rewards, c=cdict(i),label=label)

    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/cartpole/hyperparameter_plot_cartpole_nn_smoothed.png")


    

# def reinforce_baseline_or_not_nn_cartpole():
#     alpha_theta = 2**-12
#     alpha_w= 2**-9

#     parameterised_value_func_baseline = ParameterisedValueFunctionReinforce(env=gym.make("CartPole-v0"), alpha_w=alpha_w, linear=False)
#     parameterised_policy_baseline = ParameterisedPolicyReinforce(env=gym.make("CartPole-v0"), alpha_theta= alpha_theta, linear=False)
#     reinforce_agent_baseline = Reinforce(gym.make("CartPole-v0"), parameterised_policy_baseline, parameterised_value_func_baseline, baseline=True)

#     accumulated_rewards_baseline = reinforce_agent_baseline.train(500, 0.99)

#     parameterised_value_func_no_baseline = ParameterisedValueFunctionReinforce(env=gym.make("CartPole-v0"), alpha_w=alpha_w, linear=False)
#     parameterised_policy_no_baseline = ParameterisedPolicyReinforce(env=gym.make("CartPole-v0"), alpha_theta= alpha_theta, linear=False)
#     reinforce_agent_no_baseline = Reinforce(gym.make("CartPole-v0"), parameterised_policy_no_baseline, parameterised_value_func_no_baseline, baseline=False)


#     accumulated_rewards_no_baseline = reinforce_agent_no_baseline.train(500, .99)

    
#     cdict = plt.cm.get_cmap("Set2")

#     n = 10  # the larger n is, the smoother curve will be
#     b = [1.0 / n] * n
#     a = 1
#     smoothed_acc_rewards = lfilter(b,a,accumulated_rewards_baseline)
#     smoothed_acc_rewards_no_baseline = lfilter(b,a, accumulated_rewards_no_baseline)
    
#     plt.plot( np.arange(len(accumulated_rewards_baseline)),accumulated_rewards_baseline, label="with baseline", c=cdict(2), alpha=.5)
#     plt.plot( np.arange(len(accumulated_rewards_baseline)),smoothed_acc_rewards, c=cdict(2))#,label=label)

#     plt.plot( np.arange(len(accumulated_rewards_no_baseline)),accumulated_rewards_no_baseline, label="without baseline", c=cdict(0), alpha=.5)
#     plt.plot( np.arange(len(accumulated_rewards_no_baseline)),smoothed_acc_rewards_no_baseline, c=cdict(0))#,label=label)

#     plt.xlabel("Episodes")
#     plt.ylabel("Accumulated Reward per episode")
 
#     plt.legend()
#     plt.savefig("figures/cartpole/reinforce_cartpole_baseline_no.png")

def basic_reinforce_linear_cartpole():
    alpha_theta = 0.01
    alpha_w= 0.001
    env = gym.make("CartPole-v0")
    #env.observation_space.

    parameterised_policy_baseline = ParameterisedPolicyReinforce(env, alpha_theta= alpha_theta, linear=True) 
    parameterised_value_func_baseline = ParameterisedValueFunctionReinforce(env, alpha_w=alpha_w, linear=True )

    reinforce_agent_baseline = Reinforce(env, parameterised_policy_baseline, parameterised_value_func_baseline, baseline=True)

    accumulated_rewards, _ = reinforce_agent_baseline.train(500, 0.99)
    cdict = plt.cm.get_cmap("Set2")


    plt.plot( np.arange(len(accumulated_rewards)),accumulated_rewards, label="Linear REINFORCE with baseline", c=cdict(0))
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/cartpole/reinforce_baseline_cartpole_linear.png")
    plt.clf()
    
def exploration_experiment_linear_cartpole():
    alpha_theta = 0.01
    alpha_w= 0.001
    env = gym.make("CartPole-v0")
    #env.observation_space.

    parameterised_policy_baseline = ParameterisedPolicyReinforce(env, alpha_theta= alpha_theta, linear=True) 
    parameterised_value_func_baseline = ParameterisedValueFunctionReinforce(env, alpha_w=alpha_w, linear=True, M=15 )

    reinforce_agent_baseline = Reinforce(env, parameterised_policy_baseline, parameterised_value_func_baseline, baseline=True)

    accumulated_rewards, random_fractions= reinforce_agent_baseline.train(1500, 0.99)
    cdict = plt.cm.get_cmap("Set2")


    plt.plot( np.arange(len(random_fractions)),random_fractions, label="fraction of randomness per episode",c=cdict(0))
    plt.plot( np.arange(len(accumulated_rewards)),accumulated_rewards, label="accumulated reward per episode", c=cdict(1))
    
    fig, ax = plt.subplots()

    # Plot linear sequence, and set tick labels to the same color
    ax.plot(accumulated_rewards, color=cdict(0),  label="accumulated reward per episode")
    ax.tick_params(axis='y', labelcolor=cdict(0))

    # Generate a new Axes instance, on the twin-X axes (same position)
    ax2 = ax.twinx()

    # Plot exponential sequence, set scale to logarithmic and change tick color
    ax2.plot(random_fractions, color=cdict(1), label="fraction of randomness per episode")
    ax2.set_yscale('linear')
    ax2.tick_params(axis='y', labelcolor=cdict(1))

    plt.xlabel("Episodes")
 
    plt.legend()
    plt.savefig("figures/cartpole/reinforce_baseline_cartpole_linear_exploration.png")

def basic_reinforce_nn_cartpole():
    alpha_theta = 2**-12
    alpha_w= 2**-9
    env = gym.make("CartPole-v0")
    #env.observation_space.

    parameterised_policy_baseline = ParameterisedPolicyReinforce(env, alpha_theta= alpha_theta, linear=False) 
    parameterised_value_func_baseline = ParameterisedValueFunctionReinforce(env, alpha_w=alpha_w, linear=False )

    reinforce_agent_baseline = Reinforce(env, parameterised_policy_baseline, parameterised_value_func_baseline, baseline=True)

    accumulated_rewards, _ = reinforce_agent_baseline.train(500, 0.99)
    cdict = plt.cm.get_cmap("Set2")

    plt.plot( np.arange(len(accumulated_rewards)),accumulated_rewards, label="NN REINFORCE with baseline", c=cdict(0))
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/cartpole/reinforce_baseline_cartpole_nn.png")

    plt.clf()




def hyperparam_reinforce_b_linear_cartpole():
    M = [5,15,30]
    alphas_theta = [ 0.0001, 0.001, 0.01]
    alphas_w = [0.001, 0.01,0.02]

    env = gym.make("CartPole-v0")

    i=0
    df_dict = {"Mean Sum Rewards":[], "Mean Std Sum Rewards":[], "Ms":[], "alpha_thetas":[], "alpha_ws":[]}
    for m in M:
        for alpha_theta in alphas_theta:
            for alpha_w in alphas_w:
                parameterised_policy = ParameterisedPolicyReinforce(env, alpha_theta, linear=True) 
                parameterised_v = ParameterisedValueFunctionReinforce(env, alpha_w, linear=True, M=m)  # M=3, 5 worked

                reinforce_agent = Reinforce(env, parameterised_policy, parameterised_v)
                accumulated_rewards, _ = reinforce_agent.train(300)
                
                df_dict["Mean Sum Rewards"].append(np.mean(accumulated_rewards))
                df_dict["Mean Std Sum Rewards"].append(np.std(accumulated_rewards))
                df_dict["Ms"].append(m)
                df_dict["alpha_thetas"].append(alpha_theta)
                df_dict["alpha_ws"].append(alpha_w)
                i+=1
                
    df = pd.DataFrame(df_dict)
    print(df)
    df.to_csv("data/hyperparam_tuning_rb_cartpole.csv") 




######                      ACTOR CRITIC                #######

def ac_hyperparameter_plot_nn_cartpole():
    alpha_configs = [(2**-8, 2**-6),(2**-10, 2**-7),(2**-12, 2**-9), (2**-14, 2**-11), (2**-16,2**-13), (2**-18, 2**-15)]
    env = gym.make("CartPole-v0")
    cdict = plt.cm.get_cmap("Set2")

    accumulated_rewards_all = []
    
    for i, alphas in enumerate(alpha_configs):
        alpha_theta, alpha_w = alphas

        parameterised_value_func = ParameterisedValueFunctionActorCritic(env=env, alpha_w=alpha_w, linear=False)
        parameterised_policy = ParameterisedPolicyActorCritic(env, alpha_theta, False)
        one_step_ac_agent = OneStepActorCritic(env, parameterised_policy, parameterised_value_func)

        accumulated_rewards = one_step_ac_agent.train(episodes = 300, gamma=.99)
        
        label = r"$\alpha^w$,$\alpha^\theta$ = $2^{"+str(np.log2(alpha_w).astype(np.int8))+r"}, 2^{"+str(np.log2(alpha_theta).astype(np.int8))+r"}$"
        n = 10  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        smoothed_acc_rewards = lfilter(b,a,accumulated_rewards)
        
        
        plt.plot( np.arange(len(accumulated_rewards)),accumulated_rewards, label=label, c=cdict(i), alpha=.5)
        plt.plot( np.arange(len(accumulated_rewards)),smoothed_acc_rewards, c=cdict(i))#,label=label)

        accumulated_rewards_all.append(accumulated_rewards)


    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/cartpole/hyperparameter_ac_plot_cartpole_nn.png")

    plt.clf()
    
    for i,accumulated_rewards in enumerate(accumulated_rewards_all):
        alpha_w, alpha_theta = alpha_configs[i]
        label = r"$\alpha^w$,$\alpha^\theta$ = $2^{"+str(np.log2(alpha_w).astype(np.int8))+r"}, 2^{"+str(np.log2(alpha_theta).astype(np.int8))+r"}$"
        n = 10  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        smoothed_acc_rewards = lfilter(b,a,accumulated_rewards)
        plt.plot( np.arange(len(accumulated_rewards)),smoothed_acc_rewards, c=cdict(i),label=label)

    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/cartpole/hyperparameter_ac_plot_cartpole_nn_smoothed.png")



def basic_actor_critic_linear_cartpole():
    alpha_theta = 0.001 # 0.00000001,0.0000001,0.000001, 0.00001, 0.0001, 0.001 with one hot state
    alpha_w = 0.002 #0.01, 0.02, from >=0.03 diverges
    env = gym.make("CartPole-v0")
    cdict = plt.cm.get_cmap("Set2")


    parameterised_policy = ParameterisedPolicyActorCritic(env, alpha_theta, linear=True) 
    parameterised_v = ParameterisedValueFunctionActorCritic(env, alpha_w, linear=True, M=5)  # M=3, 5 worked

    actor_critic_agent = OneStepActorCritic(env, parameterised_policy, parameterised_v)
    accumulated_rewards = actor_critic_agent.train(500)

    plt.plot( np.arange(len(accumulated_rewards)),accumulated_rewards, label="Linear one step actor critic with baseline", c=cdict(0))
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/cartpole/actorcritic_cartpole_linear_basic.png")

def hyperparam_actor_critic_linear_cartpole():
    M = [5,15,30]
    alphas_theta = [ 0.00001, 0.0001, 0.001]
    alphas_w = [0.001, 0.01,0.02]

    env = gym.make("CartPole-v0")
    cdict = plt.cm.get_cmap("Set2")

    i=0
    df_dict = {"Mean Sum Rewards":[], "Mean Std Sum Rewards":[], "Ms":[], "alpha_thetas":[], "alpha_ws":[]}
    for m in M:
        for alpha_theta in alphas_theta:
            for alpha_w in alphas_w:
                parameterised_policy = ParameterisedPolicyActorCritic(env, alpha_theta, linear=True) 
                parameterised_v = ParameterisedValueFunctionActorCritic(env, alpha_w, linear=True, M=m)  # M=3, 5 worked

                actor_critic_agent = OneStepActorCritic(env, parameterised_policy, parameterised_v)
                accumulated_rewards = actor_critic_agent.train(300)
                
                df_dict["Mean Sum Rewards"].append(np.mean(accumulated_rewards))
                df_dict["Mean Std Sum Rewards"].append(np.std(accumulated_rewards))
                df_dict["Ms"].append(m)
                df_dict["alpha_thetas"].append(alpha_theta)
                df_dict["alpha_ws"].append(alpha_w)
                i+=1
                
    df = pd.DataFrame(df_dict)
    print(df)
    df.to_csv("data/hyperparam_tuning_ac_cartpole.csv") 




def main():
    np.random.seed(0)
    #torch.manual_seed(0)
    #hyperparam_actor_critic_linear_cartpole()
    #hyperparam_reinforce_b_linear_cartpole()
    #basic_reinforce_nn_cartpole()
    #ac_hyperparameter_plot_nn_cartpole()
    #exploration_experiment_linear_cartpole()
    basic_reinforce_linear_cartpole()
    basic_reinforce_nn_cartpole()
    
if __name__=="__main__":
    main()