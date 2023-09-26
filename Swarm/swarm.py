import numpy as np
import pyswarms as ps
from numba import njit
import sys
# from pyswarms.utils.functions.single_obj import sphere

print('Running on Python version: {}'.format(sys.version))

# Define the objective function
@njit
def objective_function(x):
    print(x.shape)
    # Y = np.zeros(x.shape[0])
    # for idx,y in enumerate(x):
    #     Y[idx] = ((y[0] - 1)**2 + (y[1]-2)**2 + (y[2] - 3)**2)
    # return Y
    Y = ((x[:,0] - 1)**2 + (x[:,1]-2)**2 + (x[:,2] - 3)**2)
    print(Y.shape)
    return  Y
# objective_function = sphere

# Define the bounds for each element of x
bounds = (np.array([-5]*3), np.array([5]*3))
print('Bounds:')
print(bounds)

# Define the initial guesses for each element of x
initial_guess_1 = np.array([1.0, 2.0, 2.9])

# Define the number of elements to optimize
dimensions = initial_guess_1.size
print('Dimensions:', dimensions)

# defining the number of particles to use:
n_particles = 100

# print('Objective function for initial guess:')
# print(objective_function(initial_guess_1))

# reshaping to get all the particles initial guess positions? 
# I don't know if it's necessary to do this?


initial_guess = initial_guess_1.reshape((1, dimensions))

init_pos = np.tile(initial_guess, (n_particles, 1))

# print('Initial guess of one particle:')
# print(initial_guess_1)

# print('Initial positions for all particles: ')
# print(init_pos.shape)
# print(init_pos)


# Define the options for the optimizer

options = {
    'c1': 0.5,  # cognitive parameter
    'c2': 0.3,  # social parameter
    'w': 0.9   # inertia weight
}

# Create a PSO optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, 
                                    dimensions=dimensions, 
                                    options=options, 
                                    bounds=bounds,
                                    init_pos=init_pos,
                                    ftol=1e-15,
                                    ftol_iter=100
                                    )

# Initialize the particles with the initial guesses
#optimizer.pos = init_pos

# Run the optimization
iterations = 1000
best_cost, best_position = optimizer.optimize(objective_function, iters=iterations)
best_position = np.array(best_position).reshape(1,-1)

# Print the results
print("Best position:", best_position)
print("Best cost:", best_cost)

print('Func value at best pos', objective_function(best_position))