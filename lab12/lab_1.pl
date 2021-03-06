% 12.1
% a.i.

% Butch is a killer.
% This can be represented simply by adding it as a fact
killer(butch).

% Mia and Marcellus are married.
% Again, can be represented with a simple fact
married(mia, marcellus).

% Zed is dead.
% Yet again, a fact
dead(zed).

% Marcellus kills everyone who gives Mia a foot massage.
% This is a little more complicated. The statement can be translated to
% say that Anyone who gives a foot massage to Mia will be killed by
% Marcellus, so X will represent anyone and the implied right side
% states that someone gives Mia a foot massage, while the left result is
% that Marcellus kills the person
kills(marcellus, X) :- givesFootmassage(X, mia).

% Mia loves everyone who is a good dancer.
% Similar logic to above. Given that X is a good dancer, Mia loves them
loves(mia, X) :- goodDancer(X).

% Jules eats anything that is nutritious or tasty.
% The implied side is now that a food (X) is nutritious or tasty and the
% result is that Jules eats X
eats(jules, X) :- nutritious(X); tasty(X).

% a.ii.

wizard(ron).
hasWand(harry).
quidditchPlayer(harry).
wizard(X):-  hasBroom(X),  hasWand(X).
hasBroom(X):-  quidditchPlayer(X).

% wizard(ron).
% true

% witch(ron).
% undefined procedure. witch was not a defined procedure so it could not
% determine an answer

% wizard(hermoine).
% false

% witch(hermoine).
% undefined procedure

% wizard(harry).
% true

% wizard(Y).
% Y = ron

% witch(Y).
% undefined procedure

% Prolog comes up with its answers by looking up the rule in the
% knowledge base. It's able to find the answer by the implications and
% facts in the clauses
