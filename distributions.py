"""
This file builds on the C extension _distributions.c and contains python classes
that are used to evaluate PDFs and CDFs of distributions and draw samples from
a range of distributions.

The following distributions are currently implemented
- Exponential (http://en.wikipedia.org/wiki/Exponential_distribution)
- Lognormal (http://en.wikipedia.org/wiki/Lognormal_distribution)
- Gamma (http://en.wikipedia.org/wiki/Gamma_distribution)
- Pareto (http://en.wikipedia.org/wiki/Pareto_distribution)
- Rayleigh (http://en.wikipedia.org/wiki/Rayleigh_distribution)
- Uniform (http://en.wikipedia.org/wiki/Uniform_distribution_(continuous))

Other distributions can be added easily by implementing the following template

class mydistribution:
    def __init__(self, param1, param2):
        #Initialize the parameters characterizing the distribution
        self.param1 = param1 
        self.param2 = param2
    def rvs(self):
        #Sample the distribution
        return draw_a_sample
    def pdf(self, x):
        #Evaluate the PDF at x
        return evalue_the_pdf
    def cdf(self, x):
        #Evaluate the CDF at x
        return evaluate_the_cdf
"""

from random import Random
import numpy as np
import scipy.special as special
import _distributions

random = Random()

class exponential:
    def __init__(self, mean):
        self.mean = mean
    def rvs(self):
        return random.expovariate(1 / self.mean)
    def pdf(self, x):
        return np.exp(-x / self.mean) / self.mean
    def cdf(self, x):
        return 1 - np.exp(-x / self.mean)

class lognormal:
    """
    Lognormal distribution with mean 'exp(mu + sigma ** 2 / 2)' and
    variance 'mean ** 2 * (exp(sigma ** 2) - 1)'.
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    def pdf(self, x):
        return _distributions.lognormal_pdf(self.mu, self.sigma, x)
    def cdf(self, x):
        return _distributions.lognormal_cdf(self.mu, self.sigma, x)
    def rvs(self):
        return random.lognormvariate(self.mu, self.sigma)
    @staticmethod
    def from_moments(mean, std):
        """
        This function creates a lognormal distribution with given mean and 
        standard deviation.
        """
        mean = float(mean) #Cast to floating point to avoid integer division
        sigma2 = np.log(1 + (std / mean) ** 2)
        mu = np.log(mean) - .5 * sigma2
        return lognormal(mu, np.sqrt(sigma2))

class gamma:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def rvs(self):
        return random.gammavariate(self.alpha, self.beta)
    def pdf(self, x):
        return _distributions.gamma_pdf(self.alpha, self.beta, x)
    def cdf(self, x):
        return special.gammainc(self.alpha, x / self.beta)
    def __str__(self):
        return str.format("gamma: alpha = {0}; beta = {1}", self.alpha, self.beta)
    @staticmethod
    def from_moments(mean, std):
        """
        This function creates a gamma distribution with given mean and 
        standard deviation.
        """
        return gamma((mean / std) ** 2, std ** 2 / mean)

class pareto:
    def __init__(self, k, alpha):
        self.k = k
        self.alpha = alpha
    def rvs(self):
        c = random.random()
        return self.k * (1 - c) ** (-1. / self.alpha)
    def pdf(self, x):
        return 0 if x < self.k else self.alpha * self.k ** self.alpha * \
            x ** -(self.alpha + 1)
    def cdf(self, x):
        return 0 if x < self.k else 1 - (self.k / x) ** self.alpha
    @staticmethod
    def from_moments(mean, std):
        """
        This function creates a Pareto distribution with given mean and 
        standard deviation.
        """
        var = std ** 2
        return pareto(mean + var / mean - std * np.sqrt(1 + var / mean ** 2),
                      1 + np.sqrt(1 + mean ** 2 / var))

class rayleigh:
    def __init__(self, sigma):
        self.sigma = sigma
    def pdf(self, x):
        return x / self.sigma ** 2 * np.exp(-.5 * (x / self.sigma) ** 2)
    def cdf(self, x):
        return 1 - np.exp(-.5 * (x / self.sigma) ** 2)
    def rvs(self):
        return self.sigma * np.sqrt(-2 * np.log(random.random()))

class uniform:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    def rvs(self):
        return self.lower + (self.upper - self.lower) * random.random()
    def pdf(self, x):
        return 1. / (self.upper - self.lower) if self.lower <= x <= self.upper else 0
    def cdf(self, x):
        if x > self.upper:
            return 1
        elif x < self.lower:
            return 0
        else:
            return (x - self.lower) / (self.upper - self.lower)
