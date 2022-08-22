import random

import numpy as np
import matplotlib.pyplot as plt

a_size = 10
steps = 100

p = [random.random() for _ in range(a_size)]

def reward(a):
    if not a in range(a_size):
        return

    x = random.random()
    return int(x < p[a])

v = [0] * a_size
n = [0] * a_size
g = 0

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

    if random.random() < e:
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

plt.subplot(3, 1, 1)
plt.bar(np.arange(a_size), p)

for _ in range(steps):
    step_greedy()

plt.subplot(3, 1, 2)
plt.bar(np.arange(a_size), v)

v = [0] * a_size
n = [0] * a_size

for _ in range(steps):
    step_egreedy(0.1)

plt.subplot(3, 1, 3)
plt.bar(np.arange(a_size), v)

plt.show()
