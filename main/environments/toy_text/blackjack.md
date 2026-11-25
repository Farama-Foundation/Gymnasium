# Blackjack

![image](_static/videos/toy_text/blackjack.gif)

This environment is part of the [Toy Text environments](../toy_text.md) which contains general information about the environment.

|                   |                                                  |
|-------------------|--------------------------------------------------|
| Action Space      | `Discrete(2)`                                    |
| Observation Space | `Tuple(Discrete(32), Discrete(11), Discrete(2))` |
| import            | `gymnasium.make("Blackjack-v1")`                 |

Blackjack is a card game where the goal is to beat the dealer by obtaining cards
that sum to closer to 21 (without going over 21) than the dealers cards.

## Description

The game starts with the dealer having one face up and one face down card,
while the player has two face up cards. All cards are drawn from an infinite deck
(i.e. with replacement).

The card values are:

- Face cards (Jack, Queen, King) have a point value of 10.
- Aces can either count as 11 (called a ‘usable ace’) or 1.
- Numerical cards (2-10) have a value equal to their number.

The player has the sum of cards held. The player can request
additional cards (hit) until they decide to stop (stick) or exceed 21 (bust,
immediate loss).

After the player sticks, the dealer reveals their facedown card, and draws cards
until their sum is 17 or greater. If the dealer goes bust, the player wins.

If neither the player nor the dealer busts, the outcome (win, lose, draw) is
decided by whose sum is closer to 21.

This environment corresponds to the version of the blackjack problem
described in Example 5.1 in Reinforcement Learning: An Introduction
by Sutton and Barto [<a href="#blackjack_ref">1</a>].

## Action Space

The action space is `Discrete(2)`, an integer in `{0, 1}` indicating
whether to stick or hit.

- 0: Stick
- 1: Hit

## Observation Space

The observation consists of a 3-tuple containing: the player’s current sum,
the value of the dealer’s one showing card (1-10 where 1 is ace),
and whether the player holds a usable ace (0 or 1).

The observation space is `Tuple((Discrete(32), Discrete(11), Discrete(2)))` and
observations are returned as a tuple of three integers, `(player_sum, dealer_card, usable_ace)`.

## Starting State

The starting state is initialised with the following values.

| Observation               | Values      |
|---------------------------|-------------|
| Player current sum        | 4, 5, …, 21 |
| Dealer showing card value | 1, 2, …, 10 |
| Usable Ace                | 0, 1        |

## Rewards

- win game: +1
- lose game: -1
- draw game: 0
- win game with natural blackjack:
  +1.5 (if <a href="#nat">natural</a> is True and <a href="#sab">sab</a> is False)
  +1 (otherwise)

## Episode End

The episode ends if the following happens:

- Termination:

1. The player hits and the sum of hand exceeds 21.
2. The player sticks.

An ace will always be counted as usable (11) unless it busts the player.

## Information

No additional information is returned.

## Arguments

```python
import gymnasium as gym
gym.make('Blackjack-v1', natural=False, sab=True)
```

<a id="nat"></a>`natural=False`: Whether to give an additional reward for
starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).
This only has an effect if `sab=False`, so it must be passed together with `sab=False`
when using `gym.make("Blackjack-v1")`.

<a id="sab"></a>`sab=True`: Whether to follow the exact rules outlined in the book by
Sutton and Barto. `Blackjack-v1` is registered with `sab=True` (the `BlackjackEnv` class
itself defaults to `sab=False`). If `sab` is `True`, the keyword argument `natural` will be ignored.
If the player achieves a natural blackjack and the dealer does not, the player
will win (i.e. get a reward of +1). The reverse rule does not apply.
If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

## References

<a id="blackjack_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)

## Version History

* v1: Fix the natural handling in Blackjack
* v0: Initial version release
