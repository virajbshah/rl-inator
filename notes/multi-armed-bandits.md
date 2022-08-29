# Multi-armed Bandits

> The k-armed bandit problem is a subset of the reinforcement learning problem
> where there exists but a single state. As a result, the order in which actions
> are taken does not affect the total reward as long as the same actions are
> taken. In other words, the best action is independent of the situation.

## A k-armed Bandit Problem

As above, the k-armed bandit problem allows the agent to choose one of k available
actions, to each of which can be attributed a stationary probability distribution
determining the reward signal upon taking that action. The agent chooses any of the
given actions for each time step, in attempt to maximise the total reward.

In this problem, the value function can easily be learnt as the expected reward for
any given action, by setting it to be the mean reward obtained by the agent by
taking the action.

The problem then becomes one of simply finding an optimal strategy to balance
exploration and exploitation.

## The Greedy and e-Greedy Policies

With the above described value function, the simplest policy would be to always
select the action with the highest observed value. But it is easy to see, that
once the agent arbitrarily takes an initial action, if the reward is greater
than zero, the selected action will be assigned a value greater than all others
(which have the value zero since they have never been taken). As a result, the
agent will repeatedly take the same action, only exploiting and never
exploring. This is known as the greedy policy.

A simple fix the the above issue would be to at random, decide to pick any
available action rather than only the one deemed best by the agent. This allows
the agent to explore other actions and learn a more accurate value function.
This can be achieved by taking a uniform random variable lying in the interval
[0, 1], and comparing it to a pre-selected value, e. If greater, the agent will
take the action with maximal percieved value, and otherwise will take, at
random, any action in the entire action space. This improved version of the
greedy policy is known as the e-greedy policy, and the balance between
exploitation and exploration can be manipulated by tweaking the value of e.

_Written by Viraj Shah_
