"""
This module implements local search on a simple sine function variant.
The function is not a linear function and has many max values
(see the sine function variant in graphs.py).

@author: kvlinden
@author: jde27
@version 14feb2020
"""
from search import Problem, hill_climbing, simulated_annealing, \
    exp_schedule, genetic_search
from random import randrange
import math


class SineVariant(Problem):
    """
    State: x value for the abs function variant f(x)
    Move: a new x value delta steps from the current x (in both directions) 
    """
    
    def __init__(self, initial, maximum=30.0, delta=0.001):
        self.initial = initial
        self.maximum = maximum
        self.delta = delta
        
    def actions(self, state):
        return [state + self.delta, state - self.delta]
    
    def result(self, stateIgnored, x):
        return x
    
    def value(self, x):
        return math.fabs(x * math.sin(x))


if __name__ == '__main__':

    # Formulate a problem with a 2D hill function and a single maximum value.
    maximum = 30
    initial = randrange(0, maximum)
    p = SineVariant(initial, maximum, delta=1.0)
    print('Initial                      x: ' + str(p.initial)
          + '\t\tvalue: ' + str(p.value(initial))
          )

    # Solve the problem using hill-climbing.
    hill_solution = hill_climbing(p)
    print('Hill-climbing solution       x: ' + str(hill_solution)
          + '\tvalue: ' + str(p.value(hill_solution))
          )

    # Solve the problem using simulated annealing.
    annealing_solution = simulated_annealing(
        p,
        exp_schedule(k=20, lam=0.005, limit=1000)
    )
    print('Simulated annealing solution x: ' + str(annealing_solution)
          + '\tvalue: ' + str(p.value(annealing_solution))
          )

    # Implements 100 random restarts, finding the average and best solutions for each algorithm
    best_hill_solution = 0
    best_annealing_solution = 0
    sum_hill_solutions = 0
    sum_annealing_solutions = 0
    for i in range(100):
        initial = randrange(0, maximum)
        p = SineVariant(initial, maximum, delta=1.0)
        hill_solution = p.value(hill_climbing(p))
        annealing_solution = p.value(simulated_annealing(
            p,
            exp_schedule(k=20, lam=0.005, limit=1000)
        ))
        if hill_solution > best_hill_solution:
            best_hill_solution = hill_solution
        if annealing_solution > best_annealing_solution:
            best_annealing_solution = annealing_solution
        sum_hill_solutions += hill_solution
        sum_annealing_solutions += annealing_solution

    print('\n100 Random Restarts:')
    print('\tHill-climbing average:', sum_hill_solutions/100)
    print('\tSimulated annealing average:', sum_annealing_solutions / 100)
    print('\tHill-climbing best:', best_hill_solution)
    print('\tSimulated annealing best:', best_annealing_solution)
