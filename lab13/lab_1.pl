% exercise 13.1
% a.i
directlyIn(irina, natasha).
directlyIn(natasha, olga).
directlyIn(olga, katarina).

in(X,Y) :- directlyIn(X,Y).
in(X,Y) :- directlyIn(X,Z), in(Z,Y).

% a.ii.
tran(eins,one).
tran(zwei,two).
tran(drei,three).
tran(vier,four).
tran(fuenf,five).
tran(sechs,six).
tran(sieben,seven).
tran(acht,eight).
tran(neun,nine).

listtran([],[]).
listtran([G|Tg],[E|Te]) :- tran(G,E), listtran(Tg,Te).

% b.
% Prolog does implement modus ponens
p(x).
q(x) :- p(x).

% q(x). true
