"""

I call this problem 'The Real-World Problem' in which a person has
to satisfy their basic needs by performing basic life actions (eating,
drinking, getting gas, going to work, and going to the store)

@author: Jon Ellis (jde27)
@version: 8feb2020

"""

from gps import gps
import logging

''' Solution:
initial - [hungry, has money, thirsty, low gas, no food]
drink water - [hungry, has money, quenched, low gas, no food]
get gas - [hungry, broke, quenched, car fueled, no food]
go to work - [hungry, has money, quenched, car fueled, no food]
go to store - [hungry, has money, quenched, low gas, food in pantry]
eat food - [full, has money, quenched, low gas, food in pantry]
get gas - [full, broke, quenched, car fueled, food in pantry]
go to work - [full, has money, quenched, car fueled, food in pantry]
'''
problem = {
    'initial': ['hungry', 'has money', 'thirsty', 'low gas', 'no food'],
    'goal': ['full', 'has money', 'quenched', 'car fueled', 'food in pantry'],
    'actions': [
        {
            'action': 'eat food',
            'preconds': ['food in pantry'],
            'add': ['full'],
            'delete': ['hungry']
        },
        {
            'action': 'go to work',
            'preconds': ['car fueled'],
            'add': ['has money'],
            'delete': ['broke']
        },
        {
            'action': 'get gas',
            'preconds': ['has money'],
            'add': ['car fueled', 'broke'],
            'delete': ['low gas', 'has money']
        },
        {
            'action': 'drink water',
            'preconds': [],
            'add': ['quenched'],
            'delete': ['thirsty']
        },
        {
            'action': 'go to store',
            'preconds': ['has money', 'car fueled'],
            'add': ['food in pantry', 'low gas'],
            'delete': ['no food', 'car fueled']
        }
    ]

}

if __name__ == '__main__':

    # This turns on detailed logging for the GPS "thought" process.
    # logging.basicConfig(level=logging.DEBUG)

    # Use GPS to solve the problem formulated above.
    actionSequence = gps(
        problem['initial'],
        problem['goal'],
        problem['actions']
    )

    # Print the solution, if there is one.
    if actionSequence is not None:
        for action in actionSequence:
            print(action)
    else:
        print('plan failure...')