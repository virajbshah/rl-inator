## Lecture 1: RL Introduction

Motivation
<ul>Industrial Revolution 1750-1850 and Machine Age<br>Digital Revolution 1950-now and Information Age<br>Artificial Intelligence</ul>


Can Machines Think?
Resources 

Computing Machinery and Intelligence~ A.M Turning 

**What is RL?"**
<ol>Learning by interaction with environment<br>Active
<br>Sequential<br>Goal directed<br>Learning without examples of optimal behaviour<br>Optimise Reward signals<br> Any goal can be formalised as the outcome of maximising a cumulative reward</ol>

 **Final Hypothesis**- It is science and framework of learning to make decisions from interaction


**The Interaction Loop**<br>
Interaction between an agent and an environment through action and observation

A mapping from states to action is called policy

**Core Concept**
<ul>Environment<br>Reward<br>Agent containing: <ul> Agent state<br>Policy<br>Value function estimate?<br>Model?</ul></ul>

**Agent State**

Agent
Components<ul>Agent State<br>Policy<br>Value Functions<br>Model</ul>

The history of the agent is the full sequence of observations, actions and rewards.It is used to construct the state of the agent

**Markov Decision Processes**
Useful Mathematical Framework

A decision process is Markovian is the probability of reward and subsequent state does not change upon addition of history. This means the state contains everything from the history. Doesn't mean it contains everything but adding more history won't help. Once the state is known, history can be discarded.

**Non Markovian/ Partially Observable Environment**

Agent State is a function of the History

A model predicts what the environment will do next
