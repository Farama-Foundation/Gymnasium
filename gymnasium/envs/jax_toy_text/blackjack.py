import math
import os
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
from jax import random
from jax.random import PRNGKey

from gymnasium import spaces
from gymnasium.envs.jax_toy_text.conversion import JaxEnv
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.functional import ActType, FuncEnv, StateType
from gymnasium.utils import EzPickle, seeding


RenderStateType = Tuple["pygame.Surface", str, int]  # type: ignore  # noqa: F821


deck = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])


# converts a tuple of device arrays into a tutple of ints
def obs_from_device(obs):
    return tuple([int(jax.device_get(obs[i])) for i in range(len(obs))])


def cmp(a, b):
    return (a > b).astype(int) - (a < b).astype(int)


# gets a random card(with replacement)
def random_card(key):
    key = random.split(key)[0]
    choice = random.choice(key, deck, shape=(1,))

    return choice[0].astype(int), key


# draws a starting hand of two cards
def draw_hand(key, hand):
    new_card, key = random_card(key)
    hand = hand.at[0].set(new_card)
    new_card, key = random_card(key)
    hand = hand.at[1].set(new_card)
    return hand, key


# draws a single card
def draw_card(key, hand, index):
    new_card, key = random_card(key)
    hand = hand.at[index].set(new_card)
    return key, hand, index + 1


def usable_ace(hand):  # Does this hand have a usable ace?
    return jnp.logical_and((jnp.count_nonzero(hand == 1) > 0), (sum(hand) + 10 <= 21))


# the player has decided to take a card
def take(env_state):
    state, key = env_state
    dealer_hand = state[0]
    player_hand = state[1]
    dealer_cards = state[2]
    player_cards = state[3]
    key, new_player_hand, _ = draw_card(key, player_hand, player_cards)

    return (dealer_hand, new_player_hand, dealer_cards, player_cards + 1), key


def dealer_stop(val):
    return sum_hand(val[1]) < 17


def draw_card_wrapper(val):
    return draw_card(*val)


# the player has decided to not take a card, ending the active portion
# of the game and turning control over to the dealer
def notake(env_state):
    state, key = env_state
    dealer_hand = state[0]
    player_hand = state[1]
    dealer_cards = state[2]
    player_cards = state[3]

    key, dealer_hand, dealer_cards = jax.lax.while_loop(
        dealer_stop,
        draw_card_wrapper,
        (key, dealer_hand, dealer_cards),
    )

    return (dealer_hand, player_hand, dealer_cards, player_cards), key


# gets an observation from env state
def _get_obsv(env_state):
    return (sum_hand(env_state[0][1]), env_state[0][0][0], usable_ace(env_state[0][1]))


def sum_hand(hand):  # Return current hand total
    return sum(hand) + (10 * usable_ace(hand))


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return (jnp.logical_not(is_bust(hand))) * sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return jnp.logical_and(
        jnp.logical_and(
            jnp.count_nonzero(hand) == 2, (jnp.count_nonzero(hand == 1) > 0)
        ),
        (jnp.count_nonzero(hand == 10) > 0),
    )


class BlackjackFunctional(
    FuncEnv[jnp.ndarray, jnp.ndarray, int, float, bool, RenderStateType]
):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ### Description
    Card Values:

    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.

    This game is played with an infinite deck (or with replacement).
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards.

    The player can request additional cards (hit, action=1) until they decide to stop (stick, action=0)
    or exceed 21 (bust, immediate loss).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust, the player wins.
    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    ### Action Space
    There are two actions: stick (0), and hit (1).

    ### Observation Space
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    and whether the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html).

    ### Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:

        +1.5 (if <a href="#nat">natural</a> is True)

        +1 (if <a href="#nat">natural</a> is False)

    ### Arguments

    ```
    gym.make('Jax-Blackjack-v1', natural=False, sutton_and_barto=False)
    ```

    <a id="nat">`natural=False`</a>: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

    <a id="sutton_and_barto">`sutton_and_barto=False`</a>: Whether to follow the exact rules outlined in the book by
    Sutton and Barto. If `sutton_and_barto` is `True`, the keyword argument `natural` will be ignored.
    If the player achieves a natural blackjack and the dealer does not, the player
    will win (i.e. get a reward of +1). The reverse rule does not apply.
    If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

    ### Version History
    * v1: Initial versions release (1.0.0)
    """

    action_space = spaces.Discrete(2)
    observation_space = spaces.Tuple(
        (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
    )

    # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
    # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
    natural = False

    # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
    sutton_and_barto = True

    # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
    }

    def transition(
        self, state: jnp.ndarray, action: Union[int, jnp.ndarray], key: PRNGKey
    ):

        env_state = jax.lax.cond(action, take, notake, (state, key))

        hand_state, key = env_state
        dealer_hand = hand_state[0]
        player_hand = hand_state[1]
        dealer_cards = hand_state[2]
        player_cards = hand_state[3]

        # note that only a bust or player action ends the round, the player
        # can still request another card with 21 cards
        done = (is_bust(player_hand) * action) + ((jnp.logical_not(action)) * 1)

        new_state = (dealer_hand, player_hand, dealer_cards, player_cards, done), key

        return new_state

    def initial(self, rng: PRNGKey):
        # env_state = self._reset(key)

        player_hand = jnp.zeros(21)
        dealer_hand = jnp.zeros(21)
        player_hand, rng = draw_hand(rng, player_hand)
        dealer_hand, rng = draw_hand(rng, dealer_hand)
        dealer_cards = 2
        player_cards = 2

        state = (dealer_hand, player_hand, dealer_cards, player_cards, 0)

        rng = random.split(rng)[0]

        env_state = (state, rng)

        return env_state

    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        """BlackJack observation."""
        return (sum_hand(state[1]), state[0][0], usable_ace(state[1]) * 1.0)

    def terminal(self, state: jnp.ndarray) -> jnp.ndarray:
        return (state[4]) > 0

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> jnp.ndarray:

        state = next_state

        dealer_hand = state[0]
        player_hand = state[1]

        # -1 reward if the player busts, otherwise +1 if better than dealer, 0 if tie, -1 if loss.
        reward = (
            0.0
            + (is_bust(player_hand) * -1 * action)
            + ((jnp.logical_not(action)) * cmp(score(player_hand), score(dealer_hand)))
        )

        # in the natural setting, if the player wins with a natural blackjack, then reward is 1.5
        if self.natural and not self.sutton_and_barto:
            condition = jnp.logical_and(is_natural(player_hand), (reward == 1))
            reward = reward * jnp.logical_not(condition) + 1.5 * condition

        # in the sutton and barto setting, if the player gets a natural blackjack and the dealer gets
        # a non-natural blackjack, the player wins. A dealer natural blackjack and a player
        # non-natural blackjack should result in a tie.
        if self.sutton_and_barto:
            condition = jnp.logical_and(
                is_natural(player_hand), jnp.logical_not(is_natural(dealer_hand))
            )
            reward = reward * jnp.logical_not(condition) + 1 * condition
        return reward

    def render_init(
        self, key: int, screen_width: int = 600, screen_height: int = 500
    ) -> RenderStateType:
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        rng = seeding.np_random(key)[0]

        suits = ["C", "D", "H", "S"]
        dealer_top_card_suit = rng.choice(suits)
        dealer_top_card_value_str = rng.choice(["J", "Q", "K"])

        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))

        return screen, dealer_top_card_value_str, dealer_top_card_suit

    def render_image(
        self,
        state: StateType,
        render_state: RenderStateType,
    ) -> Tuple[RenderStateType, np.ndarray]:

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy_text]`"
            )
        screen, dealer_top_card_value_str, dealer_top_card_suit = render_state

        player_sum, dealer_card_value, usable_ace = self.observation(state)
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 3
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if dealer_card_value == 1:
            display_card_value = "A"
        elif dealer_card_value == 10:
            display_card_value = dealer_top_card_value_str
        else:
            display_card_value = str(math.floor(dealer_card_value))

        screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            cwd = os.path.join(cwd, "..")
            cwd = os.path.join(cwd, "toy_text")
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            cwd = os.path.join(cwd, "..")
            cwd = os.path.join(cwd, "toy_text")
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(
            os.path.join("font", "Minecraft.ttf"), screen_height // 15
        )
        dealer_text = small_font.render(
            "Dealer: " + str(dealer_card_value), True, white
        )
        dealer_text_rect = screen.blit(dealer_text, (spacing, spacing))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_card_img = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"{dealer_top_card_suit}{display_card_value}.png",
                )
            )
        )
        dealer_card_rect = screen.blit(
            dealer_card_img,
            (
                screen_width // 2 - card_img_width - spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        screen.blit(
            hidden_card_img,
            (
                screen_width // 2 + spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        player_text = small_font.render("Player", True, white)
        player_text_rect = screen.blit(
            player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing)
        )

        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 6)
        player_sum_text = large_font.render(str(player_sum), True, white)
        player_sum_text_rect = screen.blit(
            player_sum_text,
            (
                screen_width // 2 - player_sum_text.get_width() // 2,
                player_text_rect.bottom + spacing,
            ),
        )

        if usable_ace:
            usable_ace_text = small_font.render("usable ace", True, white)
            screen.blit(
                usable_ace_text,
                (
                    screen_width // 2 - usable_ace_text.get_width() // 2,
                    player_sum_text_rect.bottom + spacing // 2,
                ),
            )
        return render_state, np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )

    def render_close(self, render_state: RenderStateType) -> None:
        """Closes the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            ) from e
        pygame.display.quit()
        pygame.quit()


class BlackJackJaxEnv(JaxEnv, EzPickle):
    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)
        env = BlackjackFunctional()
        env.transform(jax.jit)
        action_space = env.action_space
        observation_space = env.observation_space
        metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}
        super().__init__(
            env,
            observation_space=observation_space,
            action_space=action_space,
            metadata=metadata,
            render_mode=render_mode,
        )


# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)

# Jax structure inspired by https://medium.com/@ngoodger_7766/writing-an-rl-environment-in-jax-9f74338898ba


if __name__ == "__main__":

    env = BlackJackJaxEnv(render_mode="human")

    obs, info = env.reset()
    print(obs, info)
    env.render()

    terminal = False
    while not terminal:
        action = int(input("Please input an action\n"))
        obs, reward, terminal, truncated, info = env.step(action)
        print(obs, reward, terminal, truncated, info)
        env.render()

    exit()

    rng = jrng.PRNGKey(4)

    # env.transform(jax.jit)
    state, rng = env.initial(rng)
    print(state)
    exit()
    render_state = env.render_init(0)

    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state)
        render_state, _ = env.render_image(state, render_state)
        action = int(input("Please input an action"))
        # action = env.action_space.sample()
        next_state, frng = env.transition(state, action, rng)
        reward = env.reward(state, action, next_state)
        terminal = env.terminal(next_state)
        state = next_state
        print(obs)
        input()
