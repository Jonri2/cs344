'''
This module implements the Bayesian network shown in lab exercise 5.2

@author: kvlinden
@author: jde27
@version Mar 6, 2020
'''

from probability import BayesNet, enumeration_ask

# Utility variables
T, F = True, False

cancer = BayesNet([
    ('Cancer', '', 0.01),
    ('Test1', 'Cancer', {T: 0.90, F: 0.20}),
    ('Test2', 'Cancer', {T: 0.90, F: 0.20})
    ])

# Compute P(Cancer | Test1 and Test2).
# (bold)P(C | T1 ^ T2)
#   = alpha * (bold)P(C, T1, T2)
#   = alpha * (bold)P(C) * P(T1|C) * P(T2|C)
#   = alpha * <P(C) * P(T1|C) * P(T2|C), P(~C) * P(T1|~C) * P(T2|~C)>
#   = alpha * <0.01 * 0.9 * 0.9, 0.99 * 0.2 * 0.2>
#   = alpha * <0.0081, 0.0396>
#   = <0.17, 0.83>
# The probability that you don't have cancer is way higher than I would've thought. This is most likely the case since
# The probability that you have cancer at all is so low
print(enumeration_ask('Cancer', dict(Test1=T, Test2=T), cancer).show_approx())

# Compute P(Cancer | Test1 and ~Test2).
# (bold)P(C | T1 and ~T2)
#   = alpha * (bold)P(C, T1, ~T2)
#   = alpha * (bold)P(C) * P(T1|C) * P(~T2|C)
#   = alpha * <P(C) * P(T1|C) * P(~T2|C), P(~C) * P(T1|~C) * P(~T2|~C)>
#   = alpha * <0.01 * 0.9 * 0.1, 0.99 * 0.2 * 0.8>
#   = alpha * <0.0009, 0.1584>
#   = <0.006, 0.994>
# It's crazy that the probability is almost 0 that you have cancer even when 1 of the two tests was so came back
# positive. Again, the probabilities work out the way they do since P(C) is so low and having a failed test just makes
# the probability even lower.
print(enumeration_ask('Cancer', dict(Test1=T, Test2=F), cancer).show_approx())
