'''
Optimization-based Planning
An experiment into using optimization to tackle planning problems

Two costs: distance and cost integral

d(x1, x2) = || x1 - x2 ||_2
c(x1, x2) = sigma_o L_o(x_1, x_2)

where L_o(x_1, x_2) is the cost integral for one obstacle o
Let's use a logarithmic radial basis function

1/( || x - c ||_2) = L(x), where c \in R^n is the obstacle center and L(x) is the point cost

We want to integrate this over a straight line path \int L(x)
\int_0^1 1 /( || x(t) - c ||_2) dt

where x(t) is given by:
x(t) = (x2 - x1) t + x1

Put together:
\int_0^1 1 /( || (x2 - x1) t + x1 - c ||_2) dt
= \int_0^1 1 / \sqrt( ((x2 - x1) t + x1 - cx)^2 + ((y2 - y1) t + y1 - cy)^2  ) dt
u = 

# Next steps:
Test the solution to the line integral (done)
visualization matplotlib (done)
Try to differentiate wrt to the xs (done)
Set up the A matrix (done)

Add edge weight optimization
Adjust math for non-zero radius obstacles
Add support for kinodynamic paths (car curves)
Scale up the problem in size, number of obstacles, number of dimensions
'''

import numpy as np
from cost import *
from optim import *

if __name__ == '__main__':
    np.seterr('ignore') # we will be dividing by zero a lot; suppress errors
    env = Env(2.5, 2.5, epsilon = 5000)

    obs1 = RadialBarrierObstacle(1.77, 1.23, 1e-3) # 1/ x
    env.add_obstacle(obs1)

    obs2 = RadialBarrierObstacle(1.26, 1.27, weight=1e-3)     
    env.add_obstacle(obs2)
    
    obs3 = RadialBarrierObstacle(1.3, 1.8, weight=1e-3)     
    env.add_obstacle(obs3)

    # env.render2D()
    # env.render()
    test_integral()

    # Construct graph
    depth = 1
    start_node = Node(1, 1)
    goal_node = Node(2, 2)

    solveGD(start_node, goal_node, env, depth = 4)
