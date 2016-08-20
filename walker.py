"""
This file contains the main functionality of the software and is used to 
achieve two primary goals and is divided into two sections

1. Approximate the walker density as a function of time using Monte Carlo 
   simulations
2. Determine the steady state solution of a random walk on a network using the
   methodology developed in our paper
"""

from operator import itemgetter
from scipy.integrate import quad  # Use quadrature integration
import numpy as np  # General mathematics
import scipy.sparse as sparse  # Sparse representation of matrices necessary for larger networks
import scipy.sparse.linalg as linalg  # Access to eigenvectors and eigenvalues

"""
Section 1: Monte Carlo simulations

The Monte-Carlo simulations are implemented as follows

1. A large number of random walkers is allowed to make steps on a given network
2. The walker density is averaged among the walkers

The details of the Monte-Carlo simulation are explained in the appendix of our
paper.
"""

def make_step(G, origin, wtd='wtd'):
    """
    This function assumes that a walker is located on node `origin` of a network
    `G`. Furthermore, it assumes that all edges originating from `origin` have
    a data attribute `wtd` which is an instance of a waiting time distribution.
    
    This function draws a sample waiting time from all edges originating from
    `origin` and chooses the edge that gives the smallest waiting time.
    
    This function returns a tuple (`delta`, `neighbour`), where
    - `delta` is the length of time before the step was executed
    - `neighbour` is the neighbour that the step was made to from `origin`
    """
    # Optain a list of tuples (time, neighbour)
    options = [(data[wtd].rvs(), neighbour) for neighbour, data in G[origin].iteritems()]
    # Select the neighbour with the smallest waiting time
    # The argument `key = itemgetter(0)` ensures that the waiting times are compared
    return min(options, key=itemgetter(0))

def make_steps(G, origin, maxtime, wtd='wtd'):
    """
    This function assumes that a walker is located on node `origin` of a network
    `G`. Furthermore, it assumes that all edges have a data attribute `wtd` 
    which is an instance of a waiting time distribution.
    
    This function executes steps starting at `origin` until the time elapsed
    exceeds the argument `maxtime`. Only the elapsed time is constrained. The 
    number of steps executed is NOT.
    
    This function returns a list of tuples (`time`, `neighbour`), where
    - `time` is the time at which the step was executed
    - `neighbour` is the neighbour that the step was made to
    
    This function is an implementation of Algorithm 1 in Appendix 1 on page
    9 of our paper.
    """
    time = 0  # Set the current time
    current = origin  # Start with the origin
    steps = [(time, current)]  # Initialise the list of nodes
    while time <= maxtime:
        delta, node = make_step(G, current, wtd)  # Make one step
        time += delta  # Update the time...
        current = node  # ...and the node
        steps.append((time, current))  # Extend the step list
    return steps  # Return the list of steps

def steps2probability(steps, delta, bins):
    """
    This function takes a sequence of steps and computes an array representing
    the probability to find a walker on a given node in a range of time 
    intervals. The time intervals are uniformly spaced.
    
    `steps` is the sequence of step tuples obtained from, e.g. calling `make_steps`.
    `delta` is the width of a time step.
    `bins` is the number of bins.
    
    Hence, the array of probabilities corresponds to a time span 
    $[0, `delta` * `bins`]$.
    
    This function returns a dictionary keyed by node. The values are arrays of
    length `bins` whose $i^{th}% element represents the probability to find the
    walker on the associated node in the time interval 
    $[i * `delta`, (i + 1) * `delta`]$.
    
    This function is an implementation of the algorithm discussed in appendix
    2 on page 10 of our paper.
    """
    # The ith element of the vector associated with each node shall represent
    # the probability to find the walker at the respective node in the time
    # interval [i, i + 1] * delta
    probabilities = {}  # Declare a dictionary of probabilities
    # Consider all transitions
    for (t_i, _), (t_j, j) in zip(steps[1:], steps):
        p = probabilities.setdefault(j, np.zeros(bins))  # Get a default probability
        lower = int(t_j / delta)  # Index of the lowest bin involved
        upper = int(t_i / delta)  # Index of the highest bin involved
        # Did the step happen in the same bin?
        if lower == upper:
            frac = (t_i - t_j) / delta
            p[lower] += frac
        else:
            # The fractional time spent in the lower bin is given by
            # [(lower + 1) * delta - t_j] / delta and simplifying gives
            lowerfrac = lower + 1 - t_j / delta
            p[lower] += lowerfrac
            # The fractional time spent in the upper bin is given by
            # [t_i - upper * delta] / delta and simplifying gives
            upperfrac = t_i / delta - upper
            if upper < bins:
                p[upper] += upperfrac
            # The number of bins between the lower and upper bins are
            span = upper - lower - 1
            # Fill these with ones if there are bins inbetween
            if span > 0:
                p[lower + 1 : lower + 1 + span] = 1
    return probabilities

def probability_moments(probability, bins, run=0, moments=None):
    """
    This function calculates the mean and standard deviation of the probability
    to find a walker on a given node. It does so iteratively such that the 
    results of a Monte-Carlo simulation can be discarded after each simulation
    is completed.
    
    `probability` is the dictionary of probabilities obtained from `steps2probability`
    `bins` is the number of bins passed to `steps2probability`
    `run` is the number of iterations (not to be used explicitly).
    `moments` is a dictionary to keep track of moments (not to be used explicitly).
    
    This function returns a dictionary keyed by node. The value is a tuple of
    arrays. The first element is an array of means $x$ and the second array is the
    mean of $x^2$, NOT the variance.
    """
    z = np.zeros(bins)  # Create a default zero array
    probability = dict(probability)
    if moments == None: moments = {}  # Create empty dictionaries for the moments
    for node, (mean, mean2) in moments.iteritems():  # Go over each node part of the mean already
        p = probability.pop(node, z)  # Get the probability
        # Calculate the mean, mean square and update iteratively
        mean = (run * mean + p) / (run + 1)
        mean2 = (run * mean2 + p ** 2) / (run + 1)
        moments[node] = (mean, mean2)
    for node, p in probability.iteritems():  # Consider nodes that are not part of the mean already
        moments[node] = (p / (run + 1), p ** 2 / (run + 1))
    return moments
    
def walk(G, origin, bins, delta, runs, wtd='wtd', debug=False):
    """
    This function is a convenience function which calculates probability moments
    for a walker starting on node `origin` of the network `G`. The maximal
    simulation time is determined by `bins`*`delta`. The behaviour of the walker 
    is simulated `runs` times.
    
    This function assumes that each edge of the network has a data attribute
    `wtd` which is an instance of a probability distribution.
    
    `debug` is a flag which results in the run number being printed if set to True.
    
    This function returns a dictionary keyed by node. The value is a tuple of
    arrays. The first element is an array of means $x$ and the second array is the
    mean of $x^2$, NOT the variance.
    """
    maxtime = bins * delta  # The maximum time to run the simulation up to
    moments = {}  # The moments of the probability distributions
    for run in xrange(runs):
        steps = make_steps(G, origin, maxtime, wtd)  # Make enough steps
        probability = steps2probability(steps, delta, bins)  # Get the pdf
        moments = probability_moments(probability, bins, run, moments)
        if debug: 
            print run + 1  # , np.sum(probability.values()) / bins, \
                # np.sum([x for x,_ in moments.values()], axis = 0)
    return moments  # Return the moments

    
"""
Section 2: Steady state solutions
"""

def steady_state(ETM, resting_time):
    """
    This function calculates at most `k` steady state solutions of 
    a random walk on a network with effective transition matrix 
    `ETM` and mean resting time `resting_time`.
    
    Note that multiple distinct steady state solutions can exist
    if the network is not connected.
    
    This function returns a list of at most `k` tuples of the form
    (`eigenvalue`, `vector`), where `eigenvalue` is 
    the eigenvalue associated with `vector`. Steady state solutions
    will have $`eigenvalue`\approx 1$ and will be represented by
    `vector`. Note that the approximate equality is a result of
    numerical errors.  
    """
    # Get the eigenvectors as discussed in section III on page 5 of our paper
    evalues, evectors = linalg.eigs(ETM, k=1, which='LM')
    results = []
    for i in xrange(len(evalues)):
        evalue = evalues[i]  # Obtain the ith eigenvalue and eigenvector
        evector = evectors[:, i]
        # Multiply by the resting times as defined in Eq. (27) on page 6
        p = evector * resting_time
        p = p / sum(p)  # Normalise the solutions
        results.append((evalue, p))  # Add to the list of possible solutions
    results.sort(reverse=True)  # Order the results by largest eigenvalue
    return results

def ETM_rest_uniform(G, wtd, max_int=np.inf, cache=None):
    """
    This function does the same as `ETM_rest` but assumes that the
    WTDs of all edges are identical and equal to `wtd`.
    
    `cache` is a dictionary that is used to speed of the calculation
    by caching results. If cache is `None` the functionality is disabled.
    """
    n = G.number_of_nodes()
    ETM = sparse.lil_matrix((n, n))  # Set up a sparse matrix
    resting_time = np.zeros(n)  # Set up the resting times
    temp = np.zeros(n)

    for node in G:  # Go over each node
        neighbours = G.neighbors(node)  # Get all the neighbours
        n_neighbours = len(neighbours)  # Get the number of neighbours
        for neighbour in neighbours:
            temp[node] = ETM[neighbour, node] = 1. / n_neighbours
        # Check the cache
        if cache is not None and n_neighbours in cache:
            resting_time[node] = cache[n_neighbours]
        else:
            # Define the phi element
            phi = lambda t: (1 - wtd.cdf(t)) ** n_neighbours
            # Calculate the resting time
            integral = quad(phi, 0, max_int)[0]
            resting_time[node] = integral
            # Save it to the cache if desired
            if cache is not None: cache[n_neighbours] = integral

    return ETM, resting_time


def ETM_rest(G, wtd='wtd', max_int=np.inf, debug=False):
    """
    This function calculates the matrix $\mathbb{T}$ and the mean resting times
    for a network `G`. It assumes that each edge of the network has a data
    attribute `wtd` which is a probability distribution.
    
    In principle, the calculation requires the evaluation of integrals over the 
    domain $[0, \infty]$. The parameter `max_int` can be used to set a fixed
    upper limit to the integration domain if desired.
    
    The flag `debugs` indicates whether to print debug information.
    
    This function returns a tuple ($\mathbb{T}$, mean resting times).
    """
    n = G.number_of_nodes()  # The number of nodes
    ETM = sparse.lil_matrix((n, n))  # Create the effective transition matrix
    resting_time = np.zeros(n)  # Create a vector of resting times

    for node in G:
        edges = G[node].items()
        for neighbour, data in edges:
            # Define the T_{neighbour, node} matrix element
            # as in Eq. (2) on page 3 of our paper
            T = lambda t: data[wtd].pdf(t) * reduce(lambda a, b: a * b,
                [1 - data2[wtd].cdf(t) for neighbour2, data2 in edges if neighbour2 != neighbour], 1)
            # Integrate to get the effective transition matrix as in
            # Eq. (26) on page 5 of our paper
            int1 = ETM[neighbour, node] = quad(T, 0, max_int)[0]
            # Carry out the integral to find resting times as in the second column
            # of page 5
            int2 = quad(lambda t: t * T(t), 0, max_int)[0]
            # Add to the resting time on the node
            resting_time[node] += int2
        if debug: print node, int1, resting_time[node]

    return ETM, resting_time
