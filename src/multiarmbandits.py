import random

import numpy as np
import matplotlib.pyplot as plt

a_size = 10
steps = 1000
iters = 1000

p = np.random.rand(a_size)


def reward(a):
    if not a in range(a_size):
        return

    x = np.random.random()
    return int(x < p[a])


def init_ep(b=0):
    global v, n
    v = np.ones(a_size) * b
    p = np.random.rand(a_size)
    n = [0] * a_size


def step_uniform():
    a = random.choice(range(a_size))
    return reward(a)


def step_greedy():
    choices = []
    for i, a in enumerate(v):
        if a == max(v):
            choices.append(i)

    a = random.choice(choices)
    r = reward(a)

    v[a] = (v[a] * n[a] + r) / (n[a] + 1)
    n[a] += 1

    return r


def step_egreedy(e):
    choices = []

    if np.random.rand() < e:
        choices = range(a_size)
    else:
        for i, a in enumerate(v):
            if a == max(v):
                choices.append(i)

    a = random.choice(choices)
    r = reward(a)

    v[a] = (v[a] * n[a] + r) / (n[a] + 1)
    n[a] += 1

    return r


def test_policy(policy, *hparams, b=0):
    gt = np.zeros(steps)
    for _ in range(iters):
        init_ep(b)
        rt = np.array([policy(*hparams) for _ in range(steps)])
        gt += rt
    gt /= iters

    plt.plot(np.arange(steps), gt)
    plt.show()


if __name__ == '__main__':
    for i in range(1):
        plt.subplot(5, 1, 1)
        plt.bar(np.arange(a_size), p)

        plt.subplot(5, 1, 2)
        init_ep()
        rt = [step_egreedy(0) for _ in range(steps)]
        plt.bar(np.arange(a_size), v)
        print(f'00% exploration: {sum(rt)}')
        
        plt.subplot(5, 1, 3)
        init_ep()
        rt = [step_egreedy(0.1) for _ in range(steps)]
        plt.bar(np.arange(a_size), v)
        print(f'10% exploration: {sum(rt)}')
        
        plt.subplot(5, 1, 4)
        init_ep()
        rt = [step_egreedy(0.5) for _ in range(steps)]
        plt.bar(np.arange(a_size), v)
        print(f'50% exploration: {sum(rt)}')
    
        plt.subplot(5, 1, 5)
        init_ep()
        rt = [step_egreedy(0.99) for _ in range(steps)]
        plt.bar(np.arange(a_size), v)
        print(f'99% exploration: {sum(rt)}')
        
        plt.show()