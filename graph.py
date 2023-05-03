'''
graph.py
Creation and management of the Directed Acyclic Graph through the environment.
'''
import numpy as np
import copy
import matplotlib.pyplot as plt

INF = 100

class Env():
    def __init__(self, x_bound, y_bound, epsilon = 1e3):
        self.obstacle_centers = []
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.epsilon = epsilon

    def add_obstacle(self, obs):
        self.obstacle_centers.append(obs)

    def render3D(self, nodes = None):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        X = np.linspace(0, self.x_bound, self.epsilon, endpoint = False)
        Y = np.linspace(0, self.y_bound, self.epsilon, endpoint = False)

        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(X.shape)

        # Add the cost from all obstacles
        coords = np.stack((X, Y)).transpose(1, 2, 0) # 2 x H x W
        for obstacle in self.obstacle_centers:
            Z += obstacle.cost(coords)
        
        # Clip infinities so that plt does not squash smaller values
        inf = np.isinf(Z)
        too_big = Z > INF
        clip_index = np.logical_or(inf, too_big)

        Z[clip_index] = INF

        surf = ax.plot_surface(X, Y, Z, zorder = 1)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if nodes is not None:
            for node in nodes:
                for outgoing in node.outgoing:
                    xs = np.linspace(node.coord[0, 0], outgoing.dest.coord[0,0], 100)
                    ys = np.linspace(node.coord[0,1], outgoing.dest.coord[0, 1], 100)
                    cost = np.zeros((100,))
                    line = np.stack((xs, ys, cost)).T # 100 x 2
                    
                    for obstacle in self.obstacle_centers:
                        line[:, 2] += obstacle.cost(line[:, :2])
                    line[:, 2][line[:, 2] > 100] = 100
                    ax.plot(line[:, 0], line[:, 1], line[:, 2] + 3, 'r', linewidth= 4, alpha = 1, zorder =2)

        plt.show()

    def render2D(self, nodes = None):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.x_bound)
        ax.set_ylim(0, self.y_bound)
        X = np.linspace(0, self.x_bound, self.epsilon, endpoint = False)
        Y = np.linspace(0, self.y_bound, self.epsilon, endpoint = False)

        X, Y = np.meshgrid(X, Y)
        # Z = np.zeros(X.shape)

        # Add the cost from all obstacles
        # coords = np.stack((X, Y)).transpose(1, 2, 0) # 2 x H x W
        # for obstacle in self.obstacle_centers:
        #     Z += obstacle.cost(coords)
        obs = np.array([obstacle.center for obstacle in self.obstacle_centers]) # N x 2
        plt.scatter(obs[:, 0], obs[:, 1], c='b')

        # Clip infinities so that plt does not squash smaller values
        # inf = np.isinf(Z)
        # too_big = Z > INF
        # clip_index = np.logical_or(inf, too_big)

        # Z[clip_index] = INF

        # surf = ax.plot_surface(X, Y, Z, zorder = 1)
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        if nodes is not None:
            for node in nodes:
                for outgoing in node.outgoing:
                    xs = np.linspace(node.coord[0, 0], outgoing.dest.coord[0,0], 100)
                    ys = np.linspace(node.coord[0,1], outgoing.dest.coord[0, 1], 100)
                    # cost = np.zeros((100,))
                    line = np.stack((xs, ys)).T # 100 x 2
                    
                    # for obstacle in self.obstacle_centers:
                        # line[:, 2] += obstacle.cost(line[:, :2])
                    # line[:, 2][line[:, 2] > 100] = 100
                    ax.plot(line[:, 0], line[:, 1], 'r', linewidth= 1)

        plt.show() 

class Node():
    def __init__(self, cx, cy, id = -1):
        self.coord = np.array([[cx, cy]])
        self.incoming = []
        self.incoming_weights = []
        self.outgoing = []
        self.id = id

    def add_incoming(self, edge):
        self.incoming.append(edge)
        self.incoming_weights.append(0)

    def add_outgoing(self, edge):
        self.outgoing.append(edge)

    def __hash__(self):
        return hash(self.coord)
    
    def __eq__(self, other):
        return np.allclose(self.coord, other.coord)

class Edge():
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest
        self.alpha = 0

    def length_cost(self, jacobian = False):
        delta = self.dest.coord - self.source.coord
        length = np.linalg.norm(delta)
        if jacobian:
            C = np.concatenate((delta, -delta), axis = 1) / length
            return length, C
        
        return length
    
    def getCost(self, env):
        start = self.source.coord
        end = self.dest.coord
        s = np.concatenate((end, start), axis = 1)
        J = np.zeros((1, 4))
        obs_cost = 0
        for obstacle in env.obstacle_centers:
            integral, L = obstacle.integral(start, end, True)
            # system is L(x_2, x_1) ~= L([x2, x1] - [x20, x10]) + integral 
            # = L[x2, x1] + integral - L[x20, x10] 
            obs_cost += integral
            J += L
            # system is aw

        leng, C = self.length_cost(True)

        return obs_cost, J, leng, C

'''
Constructs a directed acyclic graph from start to end
We choose a diamond shaped graph for simplicity. In actuality it does not matter
how the graph is constructed, as long as we can extract paths from start to
finish.
'''
def construct_graph(depth, start_node, goal_node, env):

    FANOUT_ANGLE = np.pi / 4 # determines how wide the graph fans out to the sides
    travel_vector = goal_node.coord - start_node.coord
    travel_magnitude = np.linalg.norm(travel_vector)
    edge_len = travel_magnitude / (2 * depth) / np.cos(FANOUT_ANGLE) # length of the edges
    travel_angle = np.arctan2(travel_vector[0, 1], travel_vector[0, 0])
    left_angle = travel_angle + FANOUT_ANGLE
    left_vector = np.array([np.cos(left_angle), np.sin(left_angle)]) * edge_len
    right_angle = travel_angle - FANOUT_ANGLE
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
                left_edge = Edge(node, left)
                left.add_incoming(left_edge)
                node.add_outgoing(left_edge)
                new_frontier.append(left)
                nodes.append(left)
            else:
                edge = Edge(node, old_left)
                old_left.add_incoming(edge)
                node.add_outgoing(edge)

            # add right node
            right_coord = old_coord + right_vector
            right = Node(right_coord[0, 0], right_coord[0, 1], len(nodes) - 1)
            right_edge = Edge(node, right)
            right.add_incoming(right_edge)
            node.add_outgoing(right_edge)
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
                right_edge = Edge(node, right)
                right.add_incoming(right_edge)
                node.add_outgoing(right_edge)
                new_frontier.append(right)

            # Connect left node
            if i != 0:
                left_edge = Edge(node, old_left)
                old_left.add_incoming(left_edge)
                node.add_outgoing(left_edge)

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

    '''
    Format of the search:
    each path on the frontier will be:
    [edge, edge, ..., edge, node]
    It begins with a variable number of edges, followed by the node that the 
    final edge points to
    The edges are the series of edges to walk down to reach the current node
    from the start node
    When the path exits search, it must contain only edges
    '''
    while len(frontier) > 0:
        path = frontier.pop()
        node = path[-1]
        for outgoing in node.outgoing:
            new_path = copy.copy(path[:-1]) # exclude the node, keep the edges
            # shallow copy: each path is a list of pointers to edges. 
            # We are okay with aliasing, because we do not modify the underlying edge data structure
            new_path.append(outgoing)
            # print(f"Destination is: {outgoing}")
            if outgoing.dest == goal:
                paths.append(new_path)
            else:
                new_path.append(outgoing.dest)
                frontier.append(new_path)
    return paths

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