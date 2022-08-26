# Neural Networks:1
Neuron - a thing that holds a number 0<n<1
.It holds the greyscale value of the pixel (activation). 0 for black 1 for white.

784 neurons of image: first layer of network.<br> Last layer: 10 neurons(nos ranging from 0-9)

middle layers are to identify small details and filter accordingly. eg- Specific edges of the loop etc

Taking weighted sum basically confines image to the pixels we want in the desired region. To get the weighted sum in the range 0 to 1 we use the sigmoid function. 

Once we plug the weighted sum inot sigmoid, activation is the value of how positive the sum is. But if we want the no to light up for nos other than 0 we subtract that in the function and this subtraction term is called bias(Bias for inactivity)


# Neural Networks 2

Weights tell us the importance of each value

Cost functions: to define the difference between desired output and the output we get by calculating squares of the differences between trash and correct output. This answer is called cost which is high when innacurate. We calculate average cost over entire training data to get innacuracy

Gradient tells us what nudges to the values of cost function will decrease it the most

Gradient Descent- It is a way to converge to a local minimum.This is the vector to find the local minima which tells us how steep the descent is. To calculate this we need to compute the gradient direction, take a step downhill and reiterate. Gradient descent helps reduce value of cost function so that we obtain less random values. 

Algorithm for this is called Back Propagation

# Back Propagation

Adjustment of cost function by nudges (up or down) to get the desired outcome.<br>Size of the nudges depends on how much the difference is between the actual value and the target value.This can be done in three ways-
* Altering weight (in proportion to activation)
* Changing activation in the previous layer (in proportion to weight) [Not actually possible]
* Increasing bias

The average of the nudges for each training value forms a set of values we call negative descent.

Stochastic Training Descent- Making batches and compute each step according to these batches to calculate descent. Faster and effective
(Drunk man walking down example)

# Back Propagation Calculus
