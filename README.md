# RL-Inator

A hands-on project utilizing concepts from Reinforcement Learning, Linear Algebra, and Robotics.


## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
    * [Dependencies](#dependencies)
    * [Installations](#installations)
* [Usage Examples](#usage-examples)
* [Future Work](#future-work)
    * [Tasks](#tasks)
* [Resources and Acknowledgements](#resources-and-acknowledgements)
    * [Resources](#resources)


## About the Project

This project represents my journey in reinforcement learning.  It includes scripts implementing  
techniques to solve various problems posed in reinforcement learning ranging from solving a  
simple version of the k-armed  bandits problem to the full reinforcement learning problem, with  
environments having observation spaces large enough to appear intractable to classical tabular  
methods. In addition to these, the project contains some custom environments adhering to the  
OpenAI Gym API.


## Getting Started

### Dependencies

* [Python](https://www.python.org/)
* [Jupyter](https://jupyter.org/)
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [OpenAI Gym](https://www.gymlibrary.dev/) v0.26.1
* [OpenCV](https://opencv.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)

Versions included wherever they matter.

### Installations

Just clone this repository - <kbd>git clone https://github.com/virajbshah/rl-inator.git</kbd> - and  
you're ready to go!


## Usage Examples

Each script has its own usage. For example, the each of the Jupyter Notebooks will initialize their  
respective environments, solve them, and display their solutions upon executing all code cells.


<!-- Results and Demo -->


## Future Work

### Tasks

- [x] Go through material outlining any mathematical prerequisites, such as Linear Algebra.
- [x] Get familiarized with the core concepts behind Reinforcement Learning (Purely Theoretical).
- [x] Implement various policies for the Multi-Armed Bandits RL subproblem.
- [x] Utilize Dynamic Programming to create a general solution for problems where the MDP is known.
- [x] Solve problems with unknown MDPs using Monte Carlo methods.
- [x] Work with Temporal Difference methods (Q-Learning) to solve control problems.
- [x] Use Deep Q-Networks to adapt methods like Q-Learning to problems where using Q-Tables is not  
      feasible as a result of the complexity of the problem.
- [ ] Implement various custom OpenAI Gym environments.


## Resources and Acknowledgements

### Resources

- Introduction to Linear Algebra (Gilbert Strang)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf)
- [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
- [DeepMind x UCL Deep Learning Lecture Series](https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)
- [Foundations of Deep RL (Pieter Abbeel)](https://youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0)
- [OpenAI Gym Documentation](https://www.gymlibrary.dev/)

Maintained by Viraj Shah, as part of a program organized by [SRA VJTI](https://sravjti.in/).
