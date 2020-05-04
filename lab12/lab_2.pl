% 12.2

% a.i.
% 1. true.
% 2. false. unification is case sensitive
% 8. X = bread. unified
% 9. X = sausage, Y = bread. unified
% 14. false. X cannot be both bread as food and beer as drink

% ii.
house_elf(dobby).
witch(hermione).
witch('McGonagall').
witch(rita_skeeter).
magic(X):-  house_elf(X).
magic(X):-  wizard(X).
magic(X):-  witch(X).

% ?-  magic(house_elf).
% false. while any house_elf has magic, there is no variable house_elf
% that was defined to have magic. magic(dobby) would return true since
% he is a house_elf

% ?-  wizard(harry).
% true. a fact from lab_1

% ?-  magic(wizard).
% false. Same logic as house_elf.

% ?-  magic(’McGonagall’).
% true. McGonagall is a witch and any witch also has magic

% ?-  magic(Hermione).
% Hermione = dobby. This is some weird output. Since Hermione is entered
% with a capital letter and no quotes, it's interpretted as a variable
% (like X). It would return true if Hermione was lowecase since she is a
% witch and that's defined to have magic.

