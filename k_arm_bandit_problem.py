import random
import numpy as np


class Action:
    def __init__(self, prob, reward_system):
        self.prob = prob
        self.reward_system = reward_system


class Arm:
    def __init__(self, actions):
        self.actions = actions

    def pull(self):
        x = random.random()
        acc = 0.0
        for action in self.actions:
            acc += action.prob
            if acc > x:
                return action.reward_system()

        # should never occur
        return 0.0


class Bandit:
    def __init__(self, arms):
        self.arms = arms

    def __len__(self):
        return len(self.arms)


class Q:
    def __init__(self, alpha=0.0):
        self.qn = 0
        self.rn = 0
        self.n_tries = 0
        self.alpha = alpha

    def add_try(self, reward):
        self.qn = self.get_mean_reward()
        self.rn = reward
        self.n_tries += 1

    def is_stationary(self):
        return self.alpha == 0.0

    def get_mean_reward(self):
        if self.n_tries == 0:
            return self.rn
        else:
            if self.is_stationary():
                alpha = 1 / self.n_tries
            else:
                alpha = self.alpha
            return self.qn + (alpha * (self.rn - self.qn))


class Agent:
    def __init__(self, epsilon, strategy='simple', alpha=0.0):
        self.epsilon = epsilon
        self.strategy = strategy
        self.alpha = alpha

    def should_explore(self):
        return random.random() < self.epsilon

    @staticmethod
    def get_best_arm_index(qs):
        best_index, best_so_far = 0, qs[0].get_mean_reward()
        for index, q in enumerate(qs):
            if q.get_mean_reward() > best_so_far:
                best_index = index
        return best_index

    def simple(self, bandit, n_tries, alpha=0.0):
        steps = []
        n_arms = len(bandit)
        qs = [Q(alpha=alpha) for _ in range(n_arms)]
        reward_sum = 0.0
        for i in range(n_tries):
            # exploration vs greediness
            if self.should_explore():
                arm_index = random.randint(0, n_arms - 1)
            else:
                arm_index = self.get_best_arm_index(qs)
            best_arm_index = self.get_best_arm_index(qs)
            steps.append(qs[best_arm_index].get_mean_reward())
            reward = bandit.arms[arm_index].pull()
            qs[arm_index].add_try(reward)
            reward_sum += reward
        return reward_sum, steps

    def stationary(self, bandit, n_tries):
        return self.simple(bandit, n_tries)

    def set_alpha(self, bandit, n_tries, alpha):
        return self.simple(bandit, n_tries, alpha=alpha)

    def solve(self, bandit, n_tries):
        if self.strategy is 'stationary':
            return self.stationary(bandit, n_tries)
        elif self.strategy is 'set_alpha':
            return self.set_alpha(bandit, n_tries, self.alpha)
        return 0.0, []


def normal(reward):
    return lambda: max(0, np.random.normal(reward, 1))


def identity(reward):
    return lambda: reward


def poisson(reward):
    return lambda: np.random.poisson(reward)


def plot(steps):
    import matplotlib.pyplot as plt

    plt.plot(steps)
    plt.show()


def episode(agent, bandit_reward_f=identity):
    bandit = Bandit(
        [Arm([Action(0.5, bandit_reward_f(0.1)), Action(0.5, bandit_reward_f(0.8))]),
         Arm([Action(0.5, bandit_reward_f(0.2)), Action(0.5, bandit_reward_f(0.9))])])

    rewards, steps = agent.solve(bandit, 1000)
    plot(steps)
    return rewards


def run(strategy, epsilon):
    n = 1
    sum_r = 0.0
    for i in range(n):
        random.seed(i)
        np.random.seed(i)
        agent = Agent(epsilon, strategy=strategy, alpha=0.1)
        r = episode(agent, bandit_reward_f=identity)
        sum_r += r
    print("strategy=%s, epsilon=%.3f, mean_rewards=%.3f, n=%d" % (strategy, epsilon, (sum_r / n), n))


if __name__ == "__main__":
    for strategy in ['stationary', 'set_alpha']:
        print('epsilon strategy')
        run(strategy, 0.1)

        print('greedy strategy')
        run(strategy, 0.)
