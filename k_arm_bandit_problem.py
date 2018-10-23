import random


class Action:
    def __init__(self, prob, reward):
        self.prob = prob
        self.reward = reward


class Arm:
    def __init__(self, actions):
        self.actions = actions

    def pull(self):
        x = random.random()
        acc = 0.0
        for action in self.actions:
            acc += action.prob
            if acc > x:
                return action.reward

        # should never occur
        return 0.0


class Bandit:
    def __init__(self, arms):
        self.arms = arms

    def __len__(self):
        return len(self.arms)


class Q:
    def __init__(self, reward):
        self.rewards = reward
        self.n_tries = 0

    def add_try(self, reward):
        self.rewards += reward
        self.n_tries += 1

    def get_mean_reward(self):
        if self.n_tries == 0:
            return self.rewards
        return self.rewards / self.n_tries


class Agent:
    def __init__(self, epsilon):
        self.epsilon = epsilon

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

    def solve(self, bandit, n_tries):
        n_arms = len(bandit)
        qs = [Q(0) for _ in range(n_arms)]
        for i in range(n_tries):
            # exploration vs greediness
            if self.should_explore():
                arm_index = random.randint(0, n_arms - 1)
            else:
                arm_index = self.get_best_arm_index(qs)
            reward = bandit.arms[arm_index].pull()
            qs[arm_index].add_try(reward)

        sum_qs = sum([q.rewards for q in qs])
        return sum_qs


def play(epsilon=0.1):
    bandit = Bandit([Arm([Action(0.5, 0.1), Action(0.5, 0.8)]), Arm([Action(0.5, 0.2), Action(0.5, 0.9)])])

    agent = Agent(epsilon)

    rewards = agent.solve(bandit, 1000)
    return rewards


if __name__ == "__main__":
    n = 200

    e_range = 100
    for e in range(e_range):
        epsilon = (e + 1) / e_range
        sum_r = 0.0
        for i in range(n):
            random.seed(i)
            r = play(epsilon=epsilon)
            sum_r +=r
        print("epsilon=%.3f, mean_rewards=%.3f, n=%d" %(epsilon, (sum_r / n), n))
