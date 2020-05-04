% Exercise 12.3

witch(X) :- burn(X).
burn(wood).
burn(X) :- madeOfWood(X).
madeOfWood(X) :- floats(X).
floats(duck).
floats(X) :- equalWeight(X, duck).
equalWeight(witch, duck).

% witch(witch) = true
