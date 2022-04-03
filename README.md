# Policy Gradient Methods

As part of a project, Yugantar Prakash and I (Marios Sirtmatsis) implemented and evaluated different policy gradient methods and assembled our results in a `report`. In this repository, one can find our code and the report that include experiments using different policy gradient methods in the context of Reinforcement Learning. These experiments were conducted to show the main differences by means of performance, learning speed and other metrics between the one-step actor critic algorithm and the REINFORCE (with baseline) algorithm.

Inside the `figures` folder, one can find charts that support the conducted experiments visually. These mostly are line charts, depicting the rewards that the agent collected over time, while interacting with two different environments.

## Summary

For this project, we decided to further explore policy gradient methods and test two different ones on multiple environments. Policy gradient methods are very important and responsible for some of the biggest successes in RL. Advanced methods like A2C, A3C or DDPG showed how strong the underlying concepts of policy gradient techniques can be and with these methods, researchers were able to outperform state-of-the-art performance on multiple RL benchmarks. The goal of this work is to evaluate the REINFORCE with baseline algorithm as well as the One-Step Actor Critic on two environments. These two environments have different difficulties of getting solved and therefore, testing our implementation of the algorithms shall be tested on both to show their robustness.

## Where to go from here
To enable reproducability we chose to first start with an easy but straigtforward implementation and extend it to a more framework-like source coude from there. Therefore, one can find all conducted experiments in the different ```experiments_*.py``` files and the different ```*.ipynb```files. Inside ```actor_critic.py``` and ```reinforce.py``` one can find easy-to-reuse implementations of the two different algorithms. These rely on a certain parameterisation that can be found in ```policy_parameterisation.py``` and ```value_function_parameterisation.py```. As ellaborated in the report, we chose to use two different policy/value function parameterisations, using neural networks, as well as liner parameterisations.

Just follow the report for a detailled insight into our work.