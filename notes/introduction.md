# Introduction

> Reinforcement learning can be described as a computational method to approach
> learning from interaction.

## Reinforcement Learning

Reinforcement learning is different from other machine learning problems, in the
way that it focuses on learning, not through evaluating the end result
obtained by an agent (or model in case of other subcategories of machine learning),
but rather by taking into consideration whatever the agent observes, the actions it
takes, and their effects on the environment.

The idea is to come up with a way to find a mapping between situations and actions
that maximises a numerical reward signal over a complete episode. This involves a
certain level of planning - the agent must learn to take take actions which may
not provide the best immediate reward, but are likely to pay off over the rest of
the episode.

An important consideration is the balance between exploration and exploitation,
that is the split between the set of actions that are known to be rewarding, versus
the set of actions that may be even more rewarding, but have not been well
explored by our agent.

## Elements of Reinforcement Learning

Apart from the agent being trained and the environment it is being trained to deal
with, reinforcement learning systems typically are composed of four major elements:

- The policy followed by the agent.
- A reward function determined by the specifics of the problem.
- A value function learned by the agent.
- A model of the environment, also learned by the agent.

## Summary

Reinforcement learning is a goal-directed, interaction driven approach to learning
and automated decision making. It separates itself from different techniques by
using direct interaction with the environment as a source of information.

_Written by Viraj Shah_
