import networkx as nx
import matplotlib.pyplot as plt
import distributions
from walker import ETM_rest, steady_state, walk
import numpy as np

#Create a graph, add three nodes and edges with associated WTDs
G = nx.Graph()
G.add_edge(0, 1, wtd = distributions.exponential(1.))
G.add_edge(1, 2, wtd = distributions.uniform(.25, .75))
G.add_edge(2, 0, wtd = distributions.rayleigh(np.sqrt(2 / np.pi) / 3))

#Set up the simulation parameters
runs = 5000 #The number of Monte-Carlo simulations to execute
maxtime = 3 #The upper limit of the time interval
bins = 60 #The number of bins to divide the time interval into
delta = float(maxtime) / bins #The width of each time bin

moments = walk(G, 0, bins, delta, runs, debug = True) #Execute the simulations

time = [i * delta for i in xrange(bins)] #Create an array of time intervals

for node, (mean, mean2) in moments.iteritems(): #Iterate over all nodes
    std = np.sqrt((mean2 - mean ** 2) / (runs - 1)) #Calculate the standard dev.
    plt.errorbar(time, mean, std, label = str(node)) #Plot the data

#Calculate the total walker density and plot it as a sanity check
total = moments[0][0] + moments[1][0] + moments[2][0]
plt.plot(time, total, label = 'Total walker density')

#Calculate the effective transition matrix and mean resting time
ETM, resting_time = ETM_rest(G)

#Print the ETM and resting times
print "Effective transition matrix"
print ETM.todense()
print "Mean resting time"
print resting_time

#Calculate the steady state solution ([0] selects the solution with the largest
#eigenvector and [1] selects the eigenvector from the tuple (evalue, evector)
ss = steady_state(ETM, resting_time)[0][1]
print "Steady state solutions"
print ss
for x in ss: #Plot the steady state solution
    plt.axhline(x)

plt.legend() #Add a legend
plt.show() #Show the plot
