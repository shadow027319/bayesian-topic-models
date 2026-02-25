import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def load_ml(path):
    data = sio.loadmat(path)
    return data['A'], data['B'], data['V']


def sampleDiscrete(p, ran=None):
    """
    Helper method for sampling from an unnormalized discrete random variable using (generalized) inverse CDF sampling
    :param p: probability mass function over {0,...,num_values-1}
    :return: x \in {0,...,num_values-1} a sample drawn according to p
    """
    normalization_constant = np.sum(p)
    uniform_number = ran or np.random.rand()
    r = uniform_number * normalization_constant
    a = p[0]
    i = 0
    while a < r:
        i += 1
        a += p[i]
    return i


def barplot(W, P, T = 20, E = None, rev = False):
    """
    Function for making a sorted bar plot based on values in P, and labelling the plot with the
    corresponding names
    :param P: An array of length num_players (107)
    :param W: Array containing names of each player
    :return: None
    """
    if rev:
        return barplot_reverse(W, P, B = T, E = E)
    
    if E is not None:
        E = E[:T]
        
    plt.barh(W[:T], P[:T], xerr = E)
    # plt.yticks(np.linspace(0, T, T), labels = W)
    plt.ylim([T, -1])
    plt.xlabel('probability')
    return 
    
    
def barplot_reverse(W, P, B = 20, E = None):
    """
    Function for making a sorted bar plot based on values in P, and labelling the plot with the
    corresponding names
    :param P: An array of length num_players (107)
    :param W: Array containing names of each player
    :return: None
    """
    if E is not None:
        E = E[-B:]
        
    plt.barh(W[-B:], P[-B:], xerr = E)
    # plt.yticks(np.linspace(0, T, T), labels = W)
    plt.ylim([B, -1])
    plt.xlabel('probability')
    return 