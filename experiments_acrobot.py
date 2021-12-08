
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



def basic_reinforce_linear_acrobot():
    alpha_theta = 0.1
    alpha_w= 0.01
    env = gym.make("Acrobot-v1")
    #env.observation_space.

    parameterised_policy_baseline = ParameterisedPolicyReinforce(env, alpha_theta= alpha_theta, linear=True) 
    parameterised_value_func_baseline = ParameterisedValueFunctionReinforce(env, alpha_w=alpha_w, linear=True, M=15 )

    reinforce_agent_baseline = Reinforce(env, parameterised_policy_baseline, parameterised_value_func_baseline, baseline=True)

    accumulated_rewards, _ = reinforce_agent_baseline.train(1500, 0.99)
    cdict = plt.cm.get_cmap("Set2")
    plt.plot(accumulated_rewards, label="Linear REINFORCE with baseline", c=cdict(0))
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/acrobot/reinforce_baseline_acrobot_linear.png")
    plt.clf()

def basic_reinforce_nn_acrobot():
    alpha_theta = 2**-14
    alpha_w= 2**-11
    env = gym.make("Acrobot-v1")
    #env.observation_space.

    parameterised_policy_baseline = ParameterisedPolicyReinforce(env, alpha_theta= alpha_theta, linear=False) 
    parameterised_value_func_baseline = ParameterisedValueFunctionReinforce(env, alpha_w=alpha_w, linear=False)

    reinforce_agent_baseline = Reinforce(env, parameterised_policy_baseline, parameterised_value_func_baseline, baseline=True)

    accumulated_rewards,_ = reinforce_agent_baseline.train(1500, 0.99)

    cdict = plt.cm.get_cmap("Set2")

    plt.plot( accumulated_rewards, label="NN REINFORCE with baseline", c=cdict(0))
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward per episode")
 
    plt.legend()
    plt.savefig("figures/acrobot/reinforce_acrobot_nn.png")

    plt.clf()


def hyperparam_reinforce_b_linear_acrobot():
    M = [10,15]
    alphas_theta = [ 0.1, 0.3]
    alphas_w = [0.01,0.02]

    env = gym.make("Acrobot-v1")

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
    df.to_csv("data/hyperparam_tuning_rb_acrobot.csv") 

def reinforce_baseline_hyperparameter_plot_nn_acrobot():
    alpha_configs = [(2**-8, 2**-6),(2**-10, 2**-7),(2**-12, 2**-9), (2**-14, 2**-11), (2**-16,2**-13), (2**-18, 2**-15)]
    env = gym.make("Acrobot-v1")
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
    plt.savefig("figures/acrobot/hyperparameter_plot_acrobot_nn.png")

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
    plt.savefig("figures/acrobot/hyperparameter_plot_acrobot_nn_smoothed.png")




def main():
    torch.manual_seed(101)
    np.random.seed(0)
    #basic_reinforce_linear_acrobot()
    #basic_reinforce_nn_acrobot()
    #hyperparam_reinforce_b_linear_acrobot()
    #basic_reinforce_linear_acrobot()
    #reinforce_baseline_hyperparameter_plot_nn_acrobot()
    basic_reinforce_nn_acrobot()
if __name__=="__main__":
    main()