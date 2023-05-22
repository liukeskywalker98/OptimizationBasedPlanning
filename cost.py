import matplotlib.pyplot as plt
import numpy as np
import copy

class RadialBarrierObstacle():
    def __init__(self, cx, cy, weight = 1e-7, radius = 0):
        self.center = np.array([cx, cy])
        self.weight = weight
        self.radius = radius

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
        radius = self.radius
        # Pad the center to achieve same shape (*, 2)
        for _ in range(len(shape) - 1):
            pos = np.expand_dims(pos, axis = 0)

        return self.weight / (np.sum(np.square(coords - pos), axis = -1) - radius)
    
    '''
    This function integrates the cost in a straight line from two provided 
    endpoints.
    Input:
        start   - (N x 2) array
        end     - (N x 2) array
        jacobian - whether to return L jacobian or not

    Output:
        (N, ) array
        (N, 4) array L (optional): dL/d[x2 y2 x1 y1]
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

def length(start, end, jacobian = False):
    length = np.linalg.norm(start - end)
    delta = end - start
    if jacobian:
        C = np.concatenate((delta, -delta), axis = 1) / length
        return length, C
    
    return length