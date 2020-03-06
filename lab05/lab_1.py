'''
This module implements the Bayesian network shown in the text, Figure 14.2.
It's taken from the AIMA Python code.

@author: kvlinden
@author: jde27
@version Mar 6, 2020
'''

from probability import BayesNet, enumeration_ask, elimination_ask, rejection_sampling, likelihood_weighting, gibbs_ask

# Utility variables
T, F = True, False

# From AIMA code (probability.py) - Fig. 14.2 - burglary example
burglary = BayesNet([
    ('Burglary', '', 0.001),
    ('Earthquake', '', 0.002),
    ('Alarm', 'Burglary Earthquake', {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
    ])

# Compute P(Alarm | Burglary and ~Earthquake). <0.94, 0.06>
print(enumeration_ask('Alarm', dict(Burglary=T, Earthquake=F), burglary).show_approx())
# Compute P(John | Burglary and ~Earthquake). <0.151, 0.849>
print(enumeration_ask('JohnCalls', dict(Burglary=T, Earthquake=F), burglary).show_approx())
# Compute P(Burglary | Alarm). <0.626, 0.374>
print(enumeration_ask('Burglary', dict(Alarm=T), burglary).show_approx())
# Compute P(Burglary | John and Mary both call). <0.716, 0.284>
print(enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())

# elimination_ask() is a dynamic programming version of enumeration_ask() and is still an exact inference algorithm
print("\nElimination Ask:", elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
# Rejection sampling results vary pretty widely since there are only so many samples, but the estimate tends to be close
print("Rejection Sampling:", rejection_sampling('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary, 10000).show_approx())
# Likelihood estimate results are very similar to rejection sampling since the algorithm is similar except it doesn't
# compute the samples that violate the given clause
print("Likelihood Estimate:", likelihood_weighting('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary, 10000).show_approx())
# Gibbs sampling results vary much less than the other two algorithms and tend to be closer to the answer. This is
# because it traverses states randomly and eventually settles in a "dynamic equilibrium"
print("Gibbs Sampling:", gibbs_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
