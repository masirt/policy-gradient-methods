import numpy as np
from numpy.core.numeric import full


class Reinforce():
    def __init__(self, env, parameterised_policy, parameterised_value_func, baseline=True) -> None:
        self.env = env
        self.parameterised_policy = parameterised_policy
        self.parameterised_value_func = parameterised_value_func

        self.baseline = baseline

        
    def train(self, episodes, gamma=.99):
        accumulated_rewards = []
        all_accumulated_rewards = []


        for episode in range(episodes):
            full_trajectory = self.generate_episode()
            trajectory_rewards = [step_vals[2] for step_vals in full_trajectory]
           

            for t, step_vals in enumerate(full_trajectory):
                S_t, A_t, R_t = step_vals
               
                G = np.sum([gamma**(k-t-1)*R_k for k,
                        R_k in enumerate(trajectory_rewards[t+1:])])
                
                # print("DELTA VAL", delta_value, "DELTA POL", delta_policy)
                if self.baseline:
                    delta = self.parameterised_value_func.compute_delta(G, S_t)
                    self.parameterised_value_func.update_w(delta, S_t)

                else:
                    delta = self.parameterised_value_func.compute_delta(G,None)

                self.parameterised_policy.update_theta(delta, A_t, S_t, gamma, t)
        
            if len(accumulated_rewards) >= 100:
                accumulated_rewards[episode%len(accumulated_rewards)] = np.sum(trajectory_rewards)
            else:
                accumulated_rewards.append(np.sum(trajectory_rewards))

            all_accumulated_rewards += [np.sum(trajectory_rewards)]

            print("EPISODE: {}, SUM OF REWARDS: {}, ACC SUM REWS {}".format(
                episode, np.sum(trajectory_rewards), np.mean(accumulated_rewards)))
           
        return all_accumulated_rewards

    def generate_episode(self):
        S_t = self.env.reset()
        full_trajectory = []

        while True:

            A_t = self.parameterised_policy.determine_action(S_t)
            S_t_next, R_t, terminal, _ = self.env.step(A_t)

            full_trajectory += [(S_t, A_t, R_t)]

            if terminal:
                break

            S_t = S_t_next

        return full_trajectory

