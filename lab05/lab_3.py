'''
This module implements the Bayesian network shown in lab exercise 5.3

@author: kvlinden
@author: jde27
@version Mar 6, 2020
'''

from probability import BayesNet, enumeration_ask

# Utility variables
T, F = True, False

happy = BayesNet([
    ('Sunny', '', 0.7),
    ('Raise', '', 0.01),
    ('Happy', 'Sunny Raise', {(T, T): 1.0, (T, F): 0.7, (F, T): 0.9, (F, F): 0.1})
    ])

# Compute P(Raise | Sunny).
# (bold)P(R | S)
#   = (bold)P(R)
#   = <0.01, 0.99>
# Raise and Sunny do not have a causal relationship, so P(Raise | Sunny) = P(Raise)
print(enumeration_ask('Raise', dict(Sunny=T), happy).show_approx())

# Compute P(Raise | Happy and Sunny).
# (bold)P(R | H ^ S)
#   = alpha * (bold)P(R,H,S)
#   = alpha * (bold)P(R) * P(H|R^S) * P(S|R^H)
#   = alpha * <P(R) * P(H|R^S) P(S|R^S), P(~R) * P(H|~R^S) * P(S|~R^H)>
#   = alpha * <0.01 * 1 * 0.7, 0.99 * 0.7 * 0.7>
#   = alpha * <0.007, 0.4851>
#   = <0.014, 0.986>
# This makes sense since the reason why you're happy is most likely due to it being sunny since the probability
# of getting a raise is so low
print(enumeration_ask('Raise', dict(Happy=T, Sunny=T), happy).show_approx())

# Compute P(Raise | Happy).
# This still makes sense since the reason why you're happy is most likely due to it being sunny since the probability
# of getting a raise is so low. It's just a little more likely that the cause of being happy is a raise since it isn't
# given that it is sunny
print(enumeration_ask('Raise', dict(Happy=T), happy).show_approx())

# Compute P(Raise | Happy and ~Sunny)
# This makes sense since there's only a 0.1 probability of being happy given that it's not sunny and you didn't get a
# raise, but the probability of getting a raise is still just 0.01
print(enumeration_ask('Raise', dict(Happy=T, Sunny=F), happy).show_approx())
