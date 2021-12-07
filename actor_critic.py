import numpy as np

class OneStepActorCritic():

    def __init__(self, env,parameterised_policy, parameterised_value_func) -> None:

        self.env = env
        # self.state_space_dim = env.observation_space.shape[0]
        # self.number_of_actions = env.action_space.n
        self.parameterised_policy = parameterised_policy
        self.parameterised_value_func = parameterised_value_func

    

    def train(self, episodes, gamma = .99):
        all_accumulated_rewards = []
        accumulated_rewards = []
        for episode in range(episodes):
            S_t = self.env.reset()
            I = 1
            accumulated_reward = 0
            
            while True:
                A_t = self.parameterised_policy.determine_action(S_t)
                S_t_next, R_t, terminal, _ = self.env.step(A_t)
                accumulated_reward += R_t

                delta = self.parameterised_value_func.compute_delta(R_t, S_t, S_t_next,gamma, terminal)
                self.parameterised_value_func.update_w(delta, S_t)
                self.parameterised_policy.update_theta(delta, A_t, S_t, I)
                if terminal:
                    break
                I *= gamma
                S_t = S_t_next

            all_accumulated_rewards += [accumulated_reward]
            if len(accumulated_rewards) >= 100:
                accumulated_rewards[episode%len(accumulated_rewards)] = accumulated_reward
            else:
                accumulated_rewards.append(accumulated_reward)

            print("EPISODE: {}, SUM OF REWARDS: {}, ACC SUM REWS {}".format(
                episode, accumulated_reward, np.mean(accumulated_rewards)))

        return all_accumulated_rewards