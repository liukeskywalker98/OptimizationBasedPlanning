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
Try to differentiate wrt to the xs 
Set up the matrix
'''

import matplotlib.pyplot as plt
import numpy as np
import copy

INF = 100

class RadialBarrierObstacle():
    def __init__(self, cx, cy, weight = 1e-7):
        self.center = np.array([cx, cy])
        self.weight = weight

    '''
    This function evaluates the obstacle cost function at a given set of coords
    Input:
        coords  - (*, 2) array of coordinates
    Output:
        (*,) array of coordinates
    '''
    def cost(self, coords):
        # coords of shape (*, 2)
        shape = coords.shape
        pos = self.center
        # Pad the center to achieve same shape (*, 2)
        for _ in range(len(shape) - 1):
            pos = np.expand_dims(pos, axis = 0)

        return self.weight / np.sum(np.square(coords - pos), axis = -1)
    
    '''
    This function integrates the cost in a straight line from two provided 
    endpoints.
    Input:
        start   - (N x 2) array
        end     - (N x 2) array
        jacobian - whether to return L jacobian or not

    Output:
        (N, ) array
        (N, 4) array L (optional)
    '''
    def integral(self, start, end, jacobian = False, debug = False):
        delta = end - start #  N x 2
        to_center = start - self.center # N x 2, 2 -> N x 2
        a = delta[:, :1] # x distance from start to end point, N x 1
        c = delta[:, 1:] # y distance from start to end point, N x 1

        b = to_center[:, :1] # x distance from start point to the center
        d = to_center[:, 1:] # y distance from start point to the center
        # print(f"dx: {a}, dy: {c}, dcx: {b}, dcy: {d}")


        e = a ** 2 + c ** 2 # N x 1
        f = 2* (a * b + c * d) # N x 1
        g = b ** 2 + d ** 2 # N x 1
        if jacobian and debug:
            print(f"de numerical: {(e[1:,0] - e[0,0]) / 1e-7}")
            print(f"df numerical: {(f[1:,0] - f[0,0]) / 1e-7}")
            print(f"dg numerical: {(g[1:,0] - g[0,0]) / 1e-7}") 

        sqrt_e = np.sqrt(e) # N x 1
        determinant = 4 * e * g - f ** 2 # N x 1
        # print(f"Determinant: {determinant}")
        B = np.sqrt(determinant) # N x 1
        C = (2 * e + f) / B # N x 1
        D = f / B

        if jacobian and debug:
            print(f"dB numerical: {(B[1:,0] - B[0,0]) / 1e-7}")
            print(f"dC numerical: {(C[1:,0] - C[0,0]) / 1e-7}")
            print(f"dD numerical: {(D[1:,0] - D[0,0]) / 1e-7}") 
        arctanC = np.arctan(C)
        arctanD = np.arctan(D)

        integral1 = 2 * arctanC / B
        integral0 = 2 * arctanD / B
        def_integral = np.linalg.norm(delta, axis = -1, keepdims = True) * (integral1 - integral0)

        if jacobian:
            # return the Jacobian of the integral
            de = np.concatenate([a, c, -a, -c], axis = 1) * 2 # N x 4
            dg = np.concatenate([np.zeros(b.shape), np.zeros(b.shape), b, d], axis = 1) * 2
            df = np.concatenate([b, d, a - b, c - d], axis = 1) * 2
            if debug:
                print(f"de analytical: {de[0]}")
                print(f"df analytical: {df[0]}")
                print(f"dg analytical: {dg[0]}")

            dB = (2 * (e * dg + g * de) - f * df) / B # N x 1, N x 4 -> N x 4
            dC = (B * (2 * de + df) - (2 * e + f) * dB) / determinant
            dD = (B * df - f* dB) / determinant
            if debug:
                print(f"dB analytical: {dB[0]}")
                print(f"dC analytical: {dC[0]}")
                print(f"dD analytical: {dD[0]}")

            L = ((2 * sqrt_e * (dC / (1 + C ** 2) - dD / (1 + D ** 2)) + (arctanC - arctanD) * de / sqrt_e)/ B) - \
                (2 * sqrt_e * (arctanC - arctanD) * dB) / determinant
            return def_integral * self.weight, L * self.weight

        return def_integral * self.weight

class Env():
    def __init__(self, x_bound, y_bound, epsilon = 1e3):
        self.obstacle_centers = []
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.epsilon = epsilon

    def add_obstacle(self, obs):
        self.obstacle_centers.append(obs)

    def render(self, nodes = None):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        X = np.linspace(0, self.x_bound, self.epsilon, endpoint = False)
        Y = np.linspace(0, self.y_bound, self.epsilon, endpoint = False)

        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(X.shape)

        # Add the cost from all obstacles
        coords = np.stack((X, Y)).transpose(1, 2, 0) # 2 x H x W
        for obstacle in self.obstacle_centers:
            Z += obstacle.cost(coords)
        
        # Clip infinities so that we can view smaller values
        inf = np.isinf(Z)
        too_big = Z > INF
        clip_index = np.logical_or(inf, too_big)

        Z[clip_index] = INF

        surf = ax.plot_surface(X, Y, Z, zorder = 1)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if nodes is not None:
            for node in nodes:
                for outgoing in node.outgoing:
                    xs = np.linspace(node.coord[0, 0], outgoing.coord[0,0], 100)
                    ys = np.linspace(node.coord[0,1], outgoing.coord[0, 1], 100)
                    cost = np.zeros((100,))
                    line = np.stack((xs, ys, cost)).T # 100 x 2
                    
                    for obstacle in self.obstacle_centers:
                        line[:, 2] += obstacle.cost(line[:, :2])
                    line[:, 2][line[:, 2] > 100] = 100
                    ax.plot(line[:, 0], line[:, 1], line[:, 2] + 3, 'r', linewidth= 4, alpha = 1, zorder =2)

        plt.show() 

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

class Node():
    def __init__(self, cx, cy, id = -1):
        self.coord = np.array([[cx, cy]])
        self.incoming = []
        self.outgoing = []
        self.id = id

    def add_incoming(self, edge):
        self.incoming.append(edge)

    def add_outgoing(self, edge):
        self.outgoing.append(edge)

    def __hash__(self):
        return hash(self.coord)
    
    def __eq__(self, other):
        return np.allclose(self.coord, other.coord)

# class Edge():
#     def __init__(self, out_node, in_node):
#         self.end = out_node
#         self.start = in_node
    
def length(start, end, jacobian = False):
    length = np.linalg.norm(start - end)
    delta = end - start
    if jacobian:
        C = np.concatenate((delta, -delta), axis = 1) / length
        return length, C
    
    return length

def getCost(start, end, env):
    start = start.coord
    end = end.coord
    s = np.concatenate((end, start), axis = 1)
    J = np.zeros((1, 4))
    intercept = 0
    cost = 0
    for obstacle in env.obstacle_centers:
        integral, L = obstacle.integral(start, end, True)
        # system is L(x_2, x_1) ~= L([x2, x1] - [x20, x10]) + integral 
        # = L[x2, x1] + integral - L[x20, x10] 
        cost += integral
        J += L
        intercept += integral# - np.dot(L, s.T)
        # system is aw

    leng, C = length(start, end, True)
    cost += leng
    J += C
    intercept += leng# - np.dot(C, s.T)

    return cost, J, intercept

def commit(node_positions, nodes):
    assert node_positions.shape[0] == len(nodes) * 2 - 4
    for i in range(node_positions.shape[0] // 2):
        assert nodes[i + 1].id >= 0 
        nodes[i + 1].coord = np.expand_dims(node_positions[i*2: i*2 + 2], axis = 0)
        # print(i)

def get_node_positions(nodes):
    positions = np.zeros((len(nodes) * 2 - 4,))
    for i in range(1, len(nodes) - 1):
        positions[i * 2 - 2: i * 2] = nodes[i].coord
    return positions
    
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
                
                cost, J, intercept = getCost(start, end, env)
                
                if start_id == 2 or start_id == 4:
                    print(f"ID {start_id} had Jacobian {J[0, 2:]}")
                if end_id == 2 or end_id == 4:
                    print(f"ID {end_id} had Jacobian {J[0, :2]}")
                # For each edge, build the matrix
                if start_id >= 0:
                    A[p, start_id * 2: start_id * 2 + 2] +=  J[0, 2:]
                if end_id >= 0:
                    A[p, end_id * 2: end_id * 2 + 2] +=  J[0, :2]
                b[p] -= intercept

                # For debugging
                path_costs[p] += cost

        print("Least squares problem constructed:")
        print(A)
        # print(f"ATA: {A.T @ A}")
        # print(b)

        # Minimize the matrix - Levenberg-Marquadt Optimizer
        print("Optimizing with LM")
        lm_worked = False
        lamda = 10
        max_lm_iters = 3
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
                    
                    cost, J, intercept = getCost(start, end, env)

                    new_path_costs[p] += cost

            if np.linalg.norm(new_path_costs) < np.linalg.norm(path_costs):
                lm_worked = True
                lamda /= 2
                print(f"LM increasing trust region. Lambda: {lamda}")
            else:
                lamda *= 5
                print(f"LM reducing trust region. Trying again. Lambda: {lamda}")
            lm_iters += 1
        # Record the solution, commit the solution to the nodes
        env.render(nodes)
        iters += 1
    return

def construct_graph(depth, start_node, goal_node, env):
    travel_vector = goal_node.coord - start_node.coord
    travel_magnitude = np.linalg.norm(travel_vector)
    edge_len = travel_magnitude / (2 * depth) * np.sqrt(2) # length of the edges
    travel_angle = np.arctan2(travel_vector[0, 1], travel_vector[0, 0])
    left_angle = travel_angle + np.pi / 4
    left_vector = np.array([np.cos(left_angle), np.sin(left_angle)]) * edge_len
    right_angle = left_angle - np.pi / 2
    right_vector = np.array([np.cos(right_angle), np.sin(right_angle)]) * edge_len
    
    # All nodes
    nodes = [start_node]

    # expansionary set
    frontier = [start_node] # set of nodes that must grow new edges
    for _ in range(depth):
        new_frontier = []
        old_left = None
        for i, node in enumerate(frontier):
            # add two sucessor nodes
            old_coord = node.coord
            if i == 0:
                # add left node
                left_coord = old_coord + left_vector
                left = Node(left_coord[0, 0], left_coord[0, 1], len(nodes) - 1)
                # edge = Edge(node, left)
                left.add_incoming(node)
                node.add_outgoing(left)
                new_frontier.append(left)
                nodes.append(left)
            else:
                # edge = Edge(node, old_left)
                old_left.add_incoming(node)
                node.add_outgoing(old_left)

            # add right node
            right_coord = old_coord + right_vector
            right = Node(right_coord[0, 0], right_coord[0, 1], len(nodes) - 1)
            # edge = Edge(node, right)
            right.add_incoming(node)
            node.add_outgoing(right)
            new_frontier.append(right)

            nodes.append(right) # track the nodes

            old_left = right

        frontier = new_frontier

    # Contractionary set
    for d in range(depth):
        new_frontier = []
        old_left = None
        for i, node in enumerate(frontier):
            # add two sucessor nodes
            old_coord = node.coord
            if i < len(frontier) - 1:
                # add right node
                if d == depth - 1:
                    right = goal_node
                else:
                    right_coord = old_coord + right_vector
                    right = Node(right_coord[0, 0], right_coord[0, 1], len(nodes) - 1)
                    nodes.append(right)
                # edge = Edge(node, right)
                right.add_incoming(node)
                node.add_outgoing(right)
                new_frontier.append(right)

            # Connect left node
            if i != 0:
                # edge = Edge(node, old_left)
                old_left.add_incoming(node)
                node.add_outgoing(old_left)

            old_left = right

        frontier = new_frontier
    nodes.append(goal_node)

    # Guard against placing nodes on the obstacle
    for node in nodes:
        for obstacle in env.obstacle_centers:
            if np.allclose(node.coord[0], obstacle.center):
                node.coord += 1e-3
    return nodes
'''
Exhaustively finds all paths in the grid
'''
def search(start, goal):
    frontier = [[start]]
    paths = []
    while len(frontier) > 0:
        path = frontier.pop()
        node = path[-1]
        for outgoing in node.outgoing:
            new_path = copy.copy(path)
            new_path.append(outgoing)
            if outgoing == goal:
                paths.append(new_path)
            else:
                frontier.append(new_path)
    return paths
        

if __name__ == '__main__':
    np.seterr('ignore') # we will be dividing by zero a lot
    env = Env(3, 3, epsilon = 1000)
    obs1 = RadialBarrierObstacle(1.5, 1.5, 1e-4) # 1/ x
    env.add_obstacle(obs1)
    # env.render()
    test_integral()

    # Construct graph
    depth = 1
    start_node = Node(1, 1)
    goal_node = Node(2, 2)

    solve(start_node, goal_node, env, depth = 2)
