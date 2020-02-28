'''
This module implements a simple classroom example of probabilistic inference
over the full joint distribution specified by AIMA, Figure 13.3.
It is based on the code from AIMA probability.py.

@author: kvlinden
@author: jde27
@version Jan 1, 2013
'''

from probability import JointProbDist, enumerate_joint_ask

# The Joint Probability Distribution Fig. 13.3 (from AIMA Python)
P = JointProbDist(['Toothache', 'Cavity', 'Catch'])
T, F = True, False
P[T, T, T] = 0.108; P[T, T, F] = 0.012
P[F, T, T] = 0.072; P[F, T, F] = 0.008
P[T, F, T] = 0.016; P[T, F, F] = 0.064
P[F, F, T] = 0.144; P[F, F, F] = 0.576

# Compute P(Cavity|Catch=T)  (see the text, page 493).
# P(Cavity|Catch) = P(Cavity^Catch)/P(Catch) = (0.108 + 0.072) / (0.108 + 0.072 + 0.016 + 0.144) = 0.529
# P(~Cavity|Catch) = 1 - 0.529 = 0.471
# *bold*P(Cavity|Catch) = <0.529, 0.471>
PC = enumerate_joint_ask('Cavity', {'Catch': T}, P)
print(PC.show_approx())

# A Joint Probability Distribution for flipping 2 coins
# Heads = True, Tails = False
P = JointProbDist(['Coin1', 'Coin2'])
P[T, T] = 0.25
P[T, F] = 0.25
P[F, T] = 0.25
P[F, F] = 0.25

# *bold*P(Coin2|Coin1=heads) = <0.5, 0.5>
# This confirms that the probability of flipping a coin is 50-50 regardless of a previous flip
# Full joint distribution is generally not used in probabilistic systems because it doesn't scale well as the number of
# variables increases. It's not practical to fill a table of the size you would need for a real-world problem which has
# lots of variables
PC = enumerate_joint_ask('Coin2', {'Coin1': T}, P)
print(PC.show_approx())
