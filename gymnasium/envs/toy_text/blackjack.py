import os
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

def cmp(a, b):
    return float(a > b) - float(a < b)

"""
  Blackjack is a card game where the goal is to beat the dealer by obtaining cards
  that sum to closer to 21 (without going over 21) than the dealers cards.

  ## Description
  The game starts with the dealer having one face up and one face down card,
  while the player has two face up cards. All cards are drawn from an infinite deck
  (i.e. with replacement).

  The card values are:
  - Face cards (Jack, Queen, King) have a point value of 10.
  - Aces can either count as 11 (called a 'usable ace') or 1.
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
  The action shape is `(1,)` in the range `{0, 1}` indicating
  whether to stick or hit.

  - 0: Stick
  - 1: Hit

  ## Observation Space
  The observation consists of a 4-tuple containing: the player's current sum,
  the value of the dealer's one showing card (1-10 where 1 is ace),
  whether the player holds a usable ace (0 or 1), 
  and the true count of the table.

  The observation is returned as `(int(), int(), int(), int())`.

  ## Starting State
  The starting state is initialised with the following values.

  | Observation               | Values         |
  |---------------------------|----------------|
  | Player current sum        |  4, 5, ..., 21 |
  | Dealer showing card value |  1, 2, ..., 10 |
  | Usable Ace                |  0, 1          |

  ## Rewards
  - win game: +1
  - lose game: -1
  - draw game: 0
  - win game with natural blackjack:
  +1.5 (if <a href="#nat">natural</a> is True)
  +1 (if <a href="#nat">natural</a> is False)

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
  gym.make('Blackjack-v1', natural=False, sab=False)
  ```

  <a id="nat"></a>`natural=False`: Whether to give an additional reward for
  starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

  <a id="sab"></a>`sab=False`: Whether to follow the exact rules outlined in the book by
  Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
  If the player achieves a natural blackjack and the dealer does not, the player
  will win (i.e. get a reward of +1). The reverse rule does not apply.
  If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

  ## References
  <a id="blackjack_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
  An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)

  ## Version History
  * v1: Fix the natural handling in Blackjack
  * v0: Initial version release
  """

class BlackjackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False, num_decks=5):
        self.action_space = spaces.Discrete(4)  # 0: Stick, 1: Hit, 2: Double Down, 3: Split
        self.observation_space = spaces.Tuple(
            (spaces.MultiDiscrete([11, 11]), spaces.Discrete(11), spaces.Discrete(2), spaces.Discrete(11))
        )
        self.natural = natural
        self.sab = sab
        self.render_mode = render_mode
        self.num_decks = num_decks  # Number of decks to use
        self.deck = []  # Deck to track available cards
        self.np_random = np.random.default_rng()
        self.running_count = 0
        self.split_hands = []
        self.current_hand = []
        self._reshuffle_deck()
    
    def _reshuffle_deck(self):
        self.deck = self.num_decks * [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.np_random.shuffle(self.deck)
        self.running_count = 0
    
    def _update_running_count(self, card):
        if card in [2, 3, 4, 5, 6]:
            self.running_count += 1
        elif card in [10, 1]:
            self.running_count -= 1
    
    def _get_true_count(self):
        remaining_decks = max(1, len(self.deck) / 52)
        true_count = self.running_count / remaining_decks
        return min(5, max(-5, int(round(true_count))))
    
    def draw_card(self):
        if len(self.deck) == 0:
            self._reshuffle_deck()
        card = self.deck.pop()
        self._update_running_count(card)
        return card
    
    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def sum_hand(self, hand):
        if 1 in hand and sum(hand) + 10 <= 21:
            return sum(hand) + 10
        return sum(hand)
    
    def is_bust(self, hand):
        return self.sum_hand(hand) > 21
    
    def score(self, hand):
        return 0 if self.is_bust(hand) else self.sum_hand(hand)
    
    def is_natural(self, hand):
        return sorted(hand) == [1, 10]

    def _get_obs(self):
        player_hand = self.current_hand if len(self.current_hand) == 2 else [self.current_hand[0], 0]
        return (player_hand, self.dealer[0], int(1 in self.current_hand and sum(self.current_hand) + 10 <= 21), self._get_true_count())
    
    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit
            self.current_hand.append(self.draw_card())
            if self.is_bust(self.current_hand):
                return self._get_obs(), -1.0, True, False, {}
            return self._get_obs(), 0.0, False, False, {}

        elif action == 0:  # stick
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = cmp(self.score(self.current_hand), self.score(self.dealer))
            if self.sab and self.is_natural(self.current_hand) and not self.is_natural(self.dealer):
                reward = 1.0
            elif not self.sab and self.natural and self.is_natural(self.current_hand) and reward == 1.0:
                reward = 1.5
            return self._get_obs(), reward, True, False, {}

        elif action == 2:  # double down
            self.current_hand.append(self.draw_card())
            if self.is_bust(self.current_hand):
                return self._get_obs(), -2.0, True, False, {}
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = 2 * cmp(self.score(self.current_hand), self.score(self.dealer))
            return self._get_obs(), reward, True, False, {}
            
        elif action == 3 and len(self.current_hand) == 2 and self.current_hand[0] == self.current_hand[1]:  # split
            self.split_hands.append([self.current_hand.pop()])
            self.current_hand.append(self.draw_card())
            self.split_hands[-1].append(self.draw_card())
            return self._get_obs(), 0.0, False, False, {}
        elif action == 3:
            return self._get_obs(), -1.0, True, False, {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if len(self.deck) < 15:
            self._reshuffle_deck()
        self.dealer = self.draw_hand()
        self.current_hand = self.draw_hand()
        self.split_hands = []
        return self._get_obs(), {}


    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        player_sum, dealer_card_value, usable_ace = self._get_obs()
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 3
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(
            os.path.join("font", "Minecraft.ttf"), screen_height // 15
        )
        dealer_text = small_font.render(
            "Dealer: " + str(dealer_card_value), True, white
        )
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_card_img = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"{self.dealer_top_card_suit}{self.dealer_top_card_value_str}.png",
                )
            )
        )
        dealer_card_rect = self.screen.blit(
            dealer_card_img,
            (
                screen_width // 2 - card_img_width - spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        self.screen.blit(
            hidden_card_img,
            (
                screen_width // 2 + spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        player_text = small_font.render("Player", True, white)
        player_text_rect = self.screen.blit(
            player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing)
        )

        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 6)
        player_sum_text = large_font.render(str(player_sum), True, white)
        player_sum_text_rect = self.screen.blit(
            player_sum_text,
            (
                screen_width // 2 - player_sum_text.get_width() // 2,
                player_text_rect.bottom + spacing,
            ),
        )

        if usable_ace:
            usable_ace_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                usable_ace_text,
                (
                    screen_width // 2 - usable_ace_text.get_width() // 2,
                    player_sum_text_rect.bottom + spacing // 2,
                ),
            )
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()


# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)
