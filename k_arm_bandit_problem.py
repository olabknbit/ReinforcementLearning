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
    def __init__(self, alfa=0.0):
        self.qn = 0
        self.rn = 0
        self.n_tries = 0
        self.alfa = alfa

    def add_try(self, reward):
        self.qn = self.get_mean_reward()
        self.rn = reward
        self.n_tries += 1

    def is_stationary(self):
        return self.alfa != 0.0

    def get_mean_reward(self):
        if self.n_tries == 0:
            return self.rn
        else:
            if self.is_stationary():
                alfa = 1 / self.n_tries
            else:
                alfa = self.alfa
            return self.qn + (alfa * (self.rn - self.qn))


class Agent:
    def __init__(self, epsilon, strategy='simple', alfa=0.0):
        self.epsilon = epsilon
        self.strategy = strategy
        self.alfa = alfa

    def should_explore(self):
        if random.random() < self.epsilon:
            return True

    @staticmethod
    def get_best_arm_index(qs):
        best_index, best_so_far = 0, qs[0].get_mean_reward()
        for index, q in enumerate(qs):
            if q.get_mean_reward() > best_so_far:
                best_index = index
        return best_index

    def simple(self, bandit, n_tries, alfa=0.0):
        n_arms = len(bandit)
        qs = [Q(alfa=alfa) for _ in range(n_arms)]
        reward_sum = 0.0
        for i in range(n_tries):
            # exploration vs greediness
            if self.should_explore():
                arm_index = random.randint(0, n_arms - 1)
            else:
                arm_index = self.get_best_arm_index(qs)
            reward = bandit.arms[arm_index].pull()
            qs[arm_index].add_try(reward)
            reward_sum += reward
        return reward_sum

    def stationary(self, bandit, n_tries):
        return self.simple(bandit, n_tries)

    def not_stationary(self, bandit, n_tries, alfa):
        return self.simple(bandit, n_tries, alfa=alfa)

    def solve(self, bandit, n_tries):
        if self.strategy is 'stationary':
            return self.stationary(bandit, n_tries)
        elif self.strategy is 'not_stationary':
            return self.not_stationary(bandit, n_tries, self.alfa)
        return 0.0


def normal(reward):
    return lambda: np.random.normal(reward, 1)


def identity(reward):
    return lambda: reward


def episode(agent, bandit_reward_f):
    bandit = Bandit(
        [Arm([Action(0.5, bandit_reward_f(0.1)), Action(0.5, bandit_reward_f(0.8))]),
         Arm([Action(0.5, bandit_reward_f(0.2)), Action(0.5, bandit_reward_f(0.9))])])

    rewards = agent.solve(bandit, 1000)
    return rewards


def run(strategy):
    n = 1
    epsilon = 0.1
    sum_r = 0.0
    for i in range(n):
        random.seed(i)
        np.random.seed(i)
        agent = Agent(epsilon, strategy=strategy, alfa=0.1)
        r = episode(agent, normal)
        sum_r += r
    print("strategy=%s, epsilon=%.3f, mean_rewards=%.3f, n=%d" % (strategy, epsilon, (sum_r / n), n))


if __name__ == "__main__":
    for strategy in ['stationary', 'not_stationary']:
        run(strategy)
