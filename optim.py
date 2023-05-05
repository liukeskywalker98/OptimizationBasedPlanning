'''
optim.py
Optimization code
Runs Levenberg-Marquadt on a least squares problem for the total environmental 
and travel cost for all paths

TODO:
add in edge weights for edge optimization
add in edge weight optimization
'''
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import copy
from cost import *
from graph import *

RECORD = True
VISUAL = True
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

'''
This function uses the log sum exp trick to provide more numerical stability to
the computation of the softmax.
Inputs:
    alpha_i - exponent of the numerator term
    alpha   - array of exponents for the denominator term

Outputs:
    float value in [0, 1] representing edge weight
'''
def stable_softmax(alpha_i, alpha):
    common_factor = np.amin(alpha) # gaurds underflow
    stable_denom_exp = alpha - common_factor
    stable_numer_exp = alpha_i - common_factor
    return np.exp(stable_numer_exp) / np.sum(np.exp(stable_denom_exp))

def levenberg_marquadt(node_positions, A, b, nodes, paths, env, path_costs):
    # Minimize the matrix - Levenberg-Marquadt Optimizer
    print("Optimizing with LM")
    path_count = len(paths)

    lm_worked = False
    lamda = 1
    max_lm_iters = 9
    lm_iters = 0
    delta = 0
    EPSILON = 1e-6
    while not lm_worked and lm_iters < max_lm_iters:
        # (A^T A + lamda diag(A^T A) dx = A^T b)
        ATA = A.T @ A
        ATb = A.T @ b
        LM = ATA + lamda * np.diag(np.diag(ATA))#np.eye(ATA.shape[0])
        LM_inv = np.linalg.inv(LM)
        print(f"ATA inv: {LM_inv}")
        dx = LM_inv @ ATb

        # Check convergence
        # print(node_positions)
        # print(dx)
        commit(node_positions + dx, nodes)
        new_path_costs = np.zeros((path_count * 2))
        # Traverse each path
        for p, path in enumerate(paths):
            for i in range(len(path)):
                edge = path[i]
                
                obs_cost, leng = edge.getCost(env)
                weight = np.sqrt(edge.weight)
                obs_cost *= weight
                leng *= weight

                new_path_costs[p * 2] += obs_cost
                new_path_costs[p * 2 + 1] += leng
        delta = np.linalg.norm(new_path_costs) - np.linalg.norm(path_costs)
        if delta < 0:
            lm_worked = True
            lamda /= 2
            print(f"LM increasing trust region. Lambda: {lamda}")
        else:
            lamda *= 5
            print(f"LM reducing trust region. Trying again. Lambda: {lamda}")
        lm_iters += 1
    print(delta)
    return delta > -EPSILON

def weight_GD():
    pass

'''
USED FOR TESTING ONLY
Minimize the travel cost and environmental cost for all paths in a least square
sense. 

Use solveGD instead.

ISSUE LOG:
There are serious convergence issues with the Levenberg-Marquadt solver. The 
problem seems to be very non-smooth, and the solver tends to converge at first, 
then diverge. The implementation of the weighted losses seems to be the primary
cause, which implies an error in the equations.

A large chunk of the weight optimization math is commented out. This is because
the math is accurate for Gauss-Newton (LM with lambda = 0), but has not been
adjusted for the general case.
'''
def solve(start_node, goal_node, env, depth = 1):
    # Find total number of paths
    nodes, edges = construct_graph(depth, start_node, goal_node, env)
    paths = search(start_node, goal_node)
    # env.render3D(nodes)
    node_count = len(nodes) - 2 # excludes start and goal. Remember to index from 1 and terminate before len - 1
    path_count = len(paths)
    edge_count = len(edges)

    converged = False
    max_iters = 300
    iters = 0
    while not converged and iters < max_iters:
        print(f"Optimization Iteration {iters}")
        cost_ndim = path_count * 2
        if len(env.obstacle_centers) == 0:
            cost_ndim = path_count 
        
        A = np.zeros((cost_ndim, node_count * 2))
        dAda = np.zeros((edge_count, cost_ndim, node_count * 2))
        b = np.zeros((cost_ndim))
        dbda = np.zeros((edge_count, cost_ndim))
        path_costs = np.zeros((cost_ndim))

        # dL/da - Jacobian of loss wrt weight parameter vector a
        D = np.zeros((cost_ndim, edge_count))
        e = np.zeros((cost_ndim,))
    
        node_positions = get_node_positions(nodes)
        # Traverse each path
        for p, path in enumerate(paths):
            for i in range(len(path)):
                edge = path[i]
                start = edge.source
                end = edge.dest
                start_id = start.id
                end_id = end.id
                edge_id = edge.id
                
                obs_cost, J, leng, C = edge.getCost(env, True)

                # Edge weight
                alpha_i = edge.alpha
                alpha = np.array(end.incoming_weights)
                weight = stable_softmax(alpha_i, alpha)
                edge.weight = weight
                print(f"As a reminder: weight is {weight}")
                weight = np.sqrt(weight)

                assert 0 <= weight and weight <= 1, f"Softmax weight incorrect: {weight}"
                # J *= weight
                # C *= weight
                # obs_cost *= weight
                # leng *= weight
                dim_delta = 1
                dim_step = 2
                if len(env.obstacle_centers) == 0:
                    dim_delta = 0
                    dim_step = 1
                if start_id >= 0:
                    A[p * dim_step, start_id * 2: start_id * 2 + 2] +=  J[0, 2:] / weight
                    A[p * dim_step + dim_delta, start_id * 2: start_id * 2 + 2] +=  C[0, 2:] / weight
                    dAda[edge_id, p * dim_step, start_id * 2: start_id * 2 + 2] += J[0, 2:]
                    dAda[edge_id, p * dim_step + dim_delta, start_id * 2: start_id * 2 + 2] += C[0, 2:]
                if end_id >= 0:
                    A[p * dim_step, end_id * 2: end_id * 2 + 2] +=  J[0, :2] / weight
                    A[p * dim_step + dim_delta, end_id * 2: end_id * 2 + 2] +=  C[0, :2] / weight
                    dAda[edge_id, p * dim_step, end_id * 2: end_id * 2 + 2] += J[0, :2]
                    dAda[edge_id, p * dim_step + dim_delta, end_id * 2: end_id * 2 + 2] += C[0, :2]

                # print(f"Obs cost: {obs_cost}, Length: {leng}")
                b[p * dim_step] -= obs_cost / weight
                b[p * dim_step + dim_delta] -= leng / weight
                dbda[edge_id, p * dim_step] -= obs_cost
                dbda[edge_id, p * dim_step + dim_delta] -= leng

                path_costs[p * dim_step] += obs_cost * weight
                path_costs[p * dim_step + dim_delta] += leng * weight

        print("Least squares problem constructed:")
        # print(A)
        # print(f"ATA: {A.T @ A}")
        print(f"ATb: {A.T @ b}")
        # print(b)
        converged = levenberg_marquadt(node_positions, A, b, nodes, paths, env, path_costs)
        # env.render2D(nodes)

        # old_A = np.copy(A)
        # old_b = np.copy(b)
        # # Weight Parameter Optimization
        # # Construct dloss/dweight
        # for p, path in enumerate(paths):
        #     for i in range(len(path)):
        #         edge = path[i]
        #         end = edge.dest
        #         edge_id = edge.id
        #         start_id = start.id
        #         end_id = end.id
                
        #         obs_cost, J, leng, C = edge.getCost(env, True)

        #         weight = edge.weight

        #         assert 0 <= weight and weight <= 1, f"Softmax weight incorrect: {weight}"

        #         if start_id >= 0:
        #             A[p * 2, start_id * 2: start_id * 2 + 2] +=  J[0, 2:] * weight
        #             A[p * 2 + 1, start_id * 2: start_id * 2 + 2] +=  C[0, 2:] * weight
        #         if end_id >= 0:
        #             A[p * 2, end_id * 2: end_id * 2 + 2] +=  J[0, :2] * weight
        #             A[p * 2 + 1, end_id * 2: end_id * 2 + 2] +=  C[0, :2] * weight

        #         b[p * 2] -= obs_cost * weight
        #         b[p * 2 + 1] -= leng * weight

        #         D[2 * p, edge_id] += obs_cost
        #         D[2 * p + 1, edge_id] += leng
                
        #         e[2 * p] += obs_cost * weight
        #         e[2 * p + 1] += leng * weight

        # S = np.zeros((edge_count, edge_count))
        
        # # Construct dweight/dalpha AKA softmax Jacobian
        # for node in nodes:
        #     incoming_weights = node.incoming_weights
        #     if len(incoming_weights) == 0:
        #         continue
        #     edge_ids = np.array([edge.id for edge in node.incoming])
        #     softmax = stable_softmax(incoming_weights, incoming_weights)

        #     ndim = softmax.shape[0]
        #     softmax_matrix = np.tile(softmax, (ndim, 1))
        #     dweight_da = softmax_matrix * (np.eye(ndim) - softmax_matrix.T)
        #     r, c = np.meshgrid(edge_ids, edge_ids)
        #     S[r, c] += dweight_da
        
        # LR = 1e-2
        # # dloss/dweight * dweight/dalpha
        # dL_da_w_prime = D.T @ e # E x 1

        # dATA_da = dAda.transpose(0, 2, 1) @ old_A + old_A.T @ dAda # (E x N*2 x P*2), (P*2 x N*2) + (N*2 x P*2) (E x P*2 x N*2) = (E x N*2 x N*2)
        # ATA = old_A.T @ old_A # (N*2 x N*2)
        # ATA_inv = np.linalg.inv(ATA) # (N*2 x N*2)
        # dATA_inv_da = -ATA_inv @ dATA_da @ ATA_inv # (N*2 x N*2), (E x N*2 x N*2), (N*2 x N*2) = (E x N*2 x N*2)
        # ATb = old_A.T @ old_b # (N*2 x P*2), (P*2, 1) = (N*2 x 1)
        # dAb_da = dAda.transpose(0, 2, 1) @ old_b + old_A.T @ np.expand_dims(dbda, axis = 2) # (E x N*2 x P*2), (P*2 x 1) + (N*2 x P*2), (E x P*2 x 1) = (E x N*2 x 1)
        # ddelta_dweight = dATA_inv_da @ ATb + ATA @ dAb_da # (E x N*2 x N*2), (N*2 x 1) + (N*2 x N*2) (E x N*2 x 1) = (E x N*2 x 1)

        # dL_dw_prime = A.T @ b # (N*2 x P*2), (P*2 x 1) -> (N*2 x 1)
        # ddelta_dw = dL_dw_prime.T @ ddelta_dweight # (1 x N*2), (E x N*2 x 1) = (E x 1 x 1)
        # alpha_gradient = S.T @ (dL_da_w_prime + ddelta_dw[:, 0]) # (E,)
        # new_alphas = get_edge_alphas(edges) - LR * alpha_gradient
        # commit_weights(new_alphas, edges, nodes)

        # Record the solution, commit the solution to the nodes
        env.render2D(nodes)
        iters += 1
    return

def solveGD(start_node, goal_node, env, depth = 1):
    # Find total number of paths
    nodes, edges = construct_graph(depth, start_node, goal_node, env)
    paths = search(start_node, goal_node)
    # env.render3D(nodes)
    node_count = len(nodes) - 2 # excludes start and goal. Remember to index from 1 and terminate before len - 1
    path_count = len(paths)
    edge_count = len(edges)

    converged = False
    max_iters = 300
    LR = 1e-2
    ALPHA_LR = 3e-1
    iters = 0

    # Initialize all edge weights
    for edge in edges:
        # Edge weight
        alpha_i = edge.alpha
        alpha = np.array(edge.dest.incoming_weights)
        weight = stable_softmax(alpha_i, alpha)
        edge.weight = weight

    # Animation params
    if RECORD:
        fig, ax = plt.subplots()
        env.init_animation2D(ax, nodes)
        moviewriter = anim.FFMpegWriter()
        moviewriter.setup(fig, "optim_movie.gif", dpi=100)

    while not converged and iters < max_iters:
        print(f"Optimization Iteration {iters}")
        cost_ndim = path_count * 2
        if len(env.obstacle_centers) == 0:
            cost_ndim = path_count 
        
        A = np.zeros((cost_ndim, node_count * 2))
        b = np.zeros((cost_ndim))

        # dL/da - Jacobian of loss wrt weight parameter vector a
        D = np.zeros((cost_ndim, edge_count))
        e = np.zeros((cost_ndim,))
    
        node_positions = get_node_positions(nodes)
        # Traverse each path
        for p, path in enumerate(paths):
            for i in range(len(path)):
                edge = path[i]
                start = edge.source
                end = edge.dest
                start_id = start.id
                end_id = end.id
                edge_id = edge.id
                
                obs_cost, J, leng, C = edge.getCost(env, True)

                assert 0 <= weight and weight <= 1, f"Softmax weight incorrect: {weight}"

                dim_delta = 1
                dim_step = 2
                if len(env.obstacle_centers) == 0:
                    dim_delta = 0
                    dim_step = 1
                if start_id >= 0:
                    A[p * dim_step, start_id * 2: start_id * 2 + 2] +=  J[0, 2:] * weight
                    A[p * dim_step + dim_delta, start_id * 2: start_id * 2 + 2] +=  C[0, 2:] * weight
                if end_id >= 0:
                    A[p * dim_step, end_id * 2: end_id * 2 + 2] +=  J[0, :2] * weight
                    A[p * dim_step + dim_delta, end_id * 2: end_id * 2 + 2] +=  C[0, :2] * weight

                # print(f"Obs cost: {obs_cost}, Length: {leng}")
                b[p * dim_step] += obs_cost * weight
                b[p * dim_step + dim_delta] += leng * weight

        # print(A)
        # print(f"ATb: {A.T @ b}")
        # print(b)
        
        # Gradient Descent on the node positions
        gradient = A.T @ b / np.linalg.norm(b)
        new_positions = node_positions - LR * gradient
        commit(new_positions, nodes)

        # env.render2D(nodes)

        # Weight Parameter Optimization
        # Construct dloss/dweight
        for p, path in enumerate(paths):
            for i in range(len(path)):
                edge = path[i]
                end = edge.dest
                edge_id = edge.id
                start_id = start.id
                end_id = end.id
                
                obs_cost, J, leng, C = edge.getCost(env, True)

                weight = edge.weight

                assert 0 <= weight and weight <= 1, f"Softmax weight incorrect: {weight}"

                if start_id >= 0:
                    A[p * 2, start_id * 2: start_id * 2 + 2] +=  J[0, 2:] * weight
                    A[p * 2 + 1, start_id * 2: start_id * 2 + 2] +=  C[0, 2:] * weight
                if end_id >= 0:
                    A[p * 2, end_id * 2: end_id * 2 + 2] +=  J[0, :2] * weight
                    A[p * 2 + 1, end_id * 2: end_id * 2 + 2] +=  C[0, :2] * weight

                b[p * 2] += obs_cost * weight
                b[p * 2 + 1] += leng * weight

                D[2 * p, edge_id] += obs_cost
                D[2 * p + 1, edge_id] += leng
                
                e[2 * p] += obs_cost * weight
                e[2 * p + 1] += leng * weight

        S = np.zeros((edge_count, edge_count))
        
        # Construct dweight/dalpha AKA softmax Jacobian
        for node in nodes:
            incoming_weights = node.incoming_weights
            if len(incoming_weights) == 0:
                continue
            edge_ids = np.array([edge.id for edge in node.incoming])
            softmax = stable_softmax(incoming_weights, incoming_weights)

            ndim = softmax.shape[0]
            softmax_matrix = np.tile(softmax, (ndim, 1))
            dweight_da = softmax_matrix * (np.eye(ndim) - softmax_matrix.T)
            assert np.allclose(dweight_da, dweight_da.T)
            r, c = np.meshgrid(edge_ids, edge_ids)
            S[r, c] += dweight_da
        
        # LR = 1e-2
        # # dloss/dweight * dweight/dalpha (First term in the weight gradient)
        dL_da_w_prime = S.T @ D.T @ e / np.linalg.norm(e) # E x 1

        new_gradient = A.T @ b / np.linalg.norm(b)

        # Assemble the second term in the weight gradient
        D_plus = np.zeros((cost_ndim, edge_count))
        e_plus = np.zeros((cost_ndim,))
        D_minus = np.zeros((cost_ndim, edge_count))
        e_minus = np.zeros((cost_ndim,))
        
        epsilon = 0.01 / np.linalg.norm(new_gradient)
        position1 = node_positions + epsilon * new_gradient
        position2 = node_positions - epsilon * new_gradient
        for p, path in enumerate(paths):
            for i in range(len(path)):
                edge = path[i]
                end = edge.dest
                edge_id = edge.id
                start_id = start.id
                end_id = end.id
                
                weight = edge.weight

                assert 0 <= weight and weight <= 1, f"Softmax weight incorrect: {weight}"
                
                # First gather D+, e+ values
                commit(position1, nodes)
                obs_cost, leng = edge.getCost(env)

                D_plus[2 * p, edge_id] += obs_cost
                D_plus[2 * p + 1, edge_id] += leng
                
                e_plus[2 * p] += obs_cost * weight
                e_plus[2 * p + 1] += leng * weight

                # Then gather D-, e- values
                commit(position2, nodes)
                obs_cost, leng = edge.getCost(env)

                D_minus[2 * p, edge_id] += obs_cost
                D_minus[2 * p + 1, edge_id] += leng
                
                e_minus[2 * p] += obs_cost * weight
                e_minus[2 * p + 1] += leng * weight

        second_term = S.T @ (D_plus.T @ e_plus  / np.linalg.norm(e_plus) - D_minus.T @ e_minus / np.linalg.norm(e_minus)) / (2 * epsilon)

        alpha_gradient = dL_da_w_prime - LR * second_term
        new_alphas = get_edge_alphas(edges) - ALPHA_LR * alpha_gradient
        commit_weights(new_alphas, edges, nodes)
        # restore the node positions
        commit(new_positions, nodes)

        # Update all edge weights
        for edge in edges:
            # Edge weight
            alpha_i = edge.alpha
            alpha = np.array(edge.dest.incoming_weights)
            weight = stable_softmax(alpha_i, alpha)
            edge.weight = weight

        # Record the solution, commit the solution to the nodes
        iters += 1
        
        if RECORD:
            env.update_animation2D(nodes)
            moviewriter.grab_frame()
        elif VISUAL:
            env.render2D(nodes)
    
    path = pick_path(goal_node, start_node)
    if RECORD:
        env.update_animation2D(nodes, path)
        for _ in range(15):
            moviewriter.grab_frame()

    if RECORD:
        moviewriter.finish()

    return