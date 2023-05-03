'''
optim.py
Optimization code
'''
import matplotlib.pyplot as plt
import numpy as np
import copy
from cost import *
from graph import *

def test_integral():
    obs1 = RadialBarrierObstacle(1.5, 1.5)

    # Test our integral function
    # Try a path that will pass rather close to the singularity
    start = np.array([[1.2, 1.1]]) # 1 x 2 
    end = np.array([[2.7, 2.6]])
    samples = 150000
    xs = np.linspace(start[0, 0], end[0, 0], samples, endpoint = False)
    ys = np.linspace(start[0, 1], end[0, 1], samples, endpoint = False)
    delta = np.linalg.norm(start - end, axis = -1) / samples
    # print(delta)
    coords = np.stack((xs, ys)).T # N x 2

    point_costs = obs1.cost(coords)
    # print(point_costs)
    # print(np.amax(point_costs))
    numerical = np.sum(point_costs * delta)
    exact = obs1.integral(start, end)
    print("Comparing results of integration of cost function:")
    print(f"Numerical integration results: {numerical}")
    print(f"Analytical integration results: {exact}")

    print("Comparing results of differentiation of cost integral:")

    test_start = np.tile(start, (5,1)) # 5 x 2
    print(test_start.shape)
    assert test_start.shape == (5, 2)
    test_end = np.tile(end, (5, 1)) # 5 x 2

    finite_diff = 1e-7
    test_end[1, 0] += finite_diff
    test_end[2, 1] += finite_diff
    test_start[3, 0] += finite_diff
    test_start[4, 1] += finite_diff
    
    print(test_start)
    print(test_end)

    integral, L = obs1.integral(test_start, test_end, True)
    assert integral.shape == (5, 1) and L.shape == (5,4) #?
    approx_L = (integral[1:] - integral[0]) / finite_diff
    print(f"Numerical jacobian: {approx_L}")
    print(f"Analytical jacobian: {L[0]}")

def levenberg_marquadt(node_positions, A, b, nodes, paths, env, path_costs):
    # Minimize the matrix - Levenberg-Marquadt Optimizer
    print("Optimizing with LM")
    path_count = len(paths)

    lm_worked = False
    lamda = 1
    max_lm_iters = 6
    lm_iters = 0
    while not lm_worked and lm_iters < max_lm_iters:
        # (A^T A + lamda diag(A^T A) dx = A^T b)
        ATA = A.T @ A
        ATb = A.T @ b
        LM = ATA + lamda * np.diag(np.diag(ATA))#np.eye(ATA.shape[0])
        dx = np.linalg.inv(LM) @ ATb

        # Check convergence
        print(node_positions)
        print(dx)
        commit(node_positions + dx, nodes)
        new_path_costs = np.zeros((path_count))
        # Traverse each path
        for p, path in enumerate(paths):
            for i in range(1, len(path)):
                start = path[i - 1]
                end = path[i]
                start_id = start.id
                end_id = end.id
                
                cost, J = getCost(start, end, env)
                
                # Edge weight
            

                new_path_costs[p] += cost

        if np.linalg.norm(new_path_costs) < np.linalg.norm(path_costs):
            lm_worked = True
            lamda /= 2
            print(f"LM increasing trust region. Lambda: {lamda}")
        else:
            lamda *= 5
            print(f"LM reducing trust region. Trying again. Lambda: {lamda}")
        lm_iters += 1
    pass
'''
Want to minimize the travel cost and environmental cost
'''
def solve(start_node, goal_node, env, depth = 1):
    # Find total number of paths
    nodes = construct_graph(depth, start_node, goal_node, env)
    paths = search(start_node, goal_node)
    env.render(nodes)
    node_count = len(nodes) - 2 # excludes start and goal. Remember to index from 1 and terminate before len - 1
    path_count = len(paths)

    converged = False
    max_iters = 10
    iters = 0
    while not converged and iters < max_iters:
        print(f"Optimization Iteration {iters}")
        
        A = np.zeros((path_count, node_count * 2))
        b = np.zeros((path_count))
        path_costs = np.zeros((path_count))
    
        node_positions = get_node_positions(nodes)
        # Traverse each path
        for p, path in enumerate(paths):
            for i in range(1, len(path)):
                start = path[i - 1]
                end = path[i]
                start_id = start.id
                end_id = end.id
                
                cost, J = getCost(start, end, env)
                
                # if start_id == 2 or start_id == 4:
                #     print(f"ID {start_id} had Jacobian {J[0, 2:]}")
                # if end_id == 2 or end_id == 4:
                #     print(f"ID {end_id} had Jacobian {J[0, :2]}")
                # For each edge, build the matrix
                if start_id >= 0:
                    A[p, start_id * 2: start_id * 2 + 2] +=  J[0, 2:]
                if end_id >= 0:
                    A[p, end_id * 2: end_id * 2 + 2] +=  J[0, :2]
                b[p] -= cost

                # For debugging
                path_costs[p] += cost

        print("Least squares problem constructed:")
        print(A)
        # print(f"ATA: {A.T @ A}")
        # print(b)
        levenberg_marquadt(node_positions, A, b, nodes, paths, env, path_costs)
        
        # Record the solution, commit the solution to the nodes
        env.render(nodes)
        iters += 1
    return

'''
Get the cost of an edge (environmental and path cost)
'''
def getCost(start, end, env):
    start = start.coord
    end = end.coord
    s = np.concatenate((end, start), axis = 1)
    J = np.zeros((1, 4))
    cost = 0
    for obstacle in env.obstacle_centers:
        integral, L = obstacle.integral(start, end, True)
        # system is L(x_2, x_1) ~= L([x2, x1] - [x20, x10]) + integral 
        # = L[x2, x1] + integral - L[x20, x10] 
        cost += integral
        J += L
        # system is aw

    leng, C = length(start, end, True)
    cost += leng
    J += C

    return cost, J