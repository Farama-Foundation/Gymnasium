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
        # Ensure that player's hand is returned as a NumPy array.
        player_hand = self.current_hand if len(self.current_hand) == 2 else [self.current_hand[0], 0]
        player_hand = np.array(player_hand, dtype=np.int64)  # Convert to NumPy array
        
        return (player_hand, 
                self.dealer[0], 
                int(1 in self.current_hand and sum(self.current_hand) + 10 <= 21), 
                self._get_true_count())
    
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
            return self._get_obs(), -20.0, True, False, {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if len(self.deck) < 15:
            self._reshuffle_deck()
        self.dealer = self.draw_hand()
        self.current_hand = self.draw_hand()
        self.split_hands = []
        return self._get_obs(), {}

    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()

    def render(self):
        try:
            import pygame
        except ImportError as e:
            from gymnasium.error import DependencyNotInstalled
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install pygame`'
            ) from e

        # Set up the display dimensions.
        screen_width, screen_height = 800, 600
        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
            self.clock = pygame.time.Clock()
        else:
            screen_width, screen_height = self.screen.get_size()

        # Define colors and fonts.
        bg_color = (0, 128, 0)        # dark green background
        card_color = (255, 255, 255)  # white card background
        border_color = (0, 0, 0)      # black border
        text_color = (255, 255, 255)  # white text

        self.screen.fill(bg_color)

        pygame.font.init()
        font_small = pygame.font.SysFont("Arial", 20)
        font_medium = pygame.font.SysFont("Arial", 24)
        font_large = pygame.font.SysFont("Arial", 32)

        # Helper functions for drawing cards.
        def draw_card(surface, card, x, y, width, height):
            rect = pygame.Rect(x, y, width, height)
            pygame.draw.rect(surface, card_color, rect)
            pygame.draw.rect(surface, border_color, rect, 2)
            label = "A" if card == 1 else str(card)
            text_surf = font_medium.render(label, True, border_color)
            text_rect = text_surf.get_rect(center=rect.center)
            surface.blit(text_surf, text_rect)

        def draw_hidden_card(surface, x, y, width, height):
            rect = pygame.Rect(x, y, width, height)
            pygame.draw.rect(surface, (50, 50, 50), rect)  # dark gray for the back
            pygame.draw.rect(surface, border_color, rect, 2)
            text_surf = font_medium.render("?", True, border_color)
            text_rect = text_surf.get_rect(center=rect.center)
            surface.blit(text_surf, text_rect)

        # Card dimensions and spacing.
        card_width = 60
        card_height = 90
        spacing = 10

        # --- Draw the Dealer's Hand ---
        dealer_title = font_large.render("Dealer's Hand", True, text_color)
        self.screen.blit(dealer_title, (spacing, spacing))
        dealer_x = spacing
        dealer_y = spacing + dealer_title.get_height() + spacing

        # Hide the dealer's second card if only two cards are present.
        hide_dealer_second = (len(self.dealer) == 2)
        for i, card in enumerate(self.dealer):
            x = dealer_x + i * (card_width + spacing)
            if hide_dealer_second and i == 1:
                draw_hidden_card(self.screen, x, dealer_y, card_width, card_height)
            else:
                draw_card(self.screen, card, x, dealer_y, card_width, card_height)

        # --- Draw the Player's Hand(s) ---
        if len(self.split_hands) == 0:
            player_title = font_large.render("Player's Hand", True, text_color)
            player_title_y = dealer_y + card_height + 2 * spacing
            self.screen.blit(player_title, (spacing, player_title_y))
            player_x = spacing
            player_y = player_title_y + player_title.get_height() + spacing
            for i, card in enumerate(self.current_hand):
                x = player_x + i * (card_width + spacing)
                draw_card(self.screen, card, x, player_y, card_width, card_height)
            # Display the player's hand total.
            total = self.sum_hand(self.current_hand)
            total_text = font_medium.render(f"Total: {total}", True, text_color)
            self.screen.blit(total_text, (player_x, player_y + card_height + spacing))
        else:
            hands_title = font_large.render("Player's Hands", True, text_color)
            hands_title_y = dealer_y + card_height + 2 * spacing
            self.screen.blit(hands_title, (spacing, hands_title_y))
            start_y = hands_title_y + hands_title.get_height() + spacing
            all_hands = [self.current_hand] + self.split_hands
            for j, hand in enumerate(all_hands):
                hand_label = font_medium.render(f"Hand {j+1}:", True, text_color)
                hand_y = start_y + j * (card_height + 3 * spacing)
                self.screen.blit(hand_label, (spacing, hand_y))
                hand_x = spacing + hand_label.get_width() + spacing
                for i, card in enumerate(hand):
                    x = hand_x + i * (card_width + spacing)
                    draw_card(self.screen, card, x, hand_y, card_width, card_height)
                total = self.sum_hand(hand)
                total_text = font_medium.render(f"Total: {total}", True, text_color)
                self.screen.blit(total_text, (hand_x, hand_y + card_height + spacing))

        # --- Render Card Counts in a Panel on the Right ---
        # Calculate counts for each card rank (Ace=1 through 10)
        card_counts = {i: self.deck.count(i) for i in range(1, 11)}
        panel_width = 150  # width of the counts panel
        panel_x = screen_width - panel_width - spacing  # position the panel on the far right
        panel_y = spacing  # start near the top

        # Draw a background for the panel (optional)
        panel_rect = pygame.Rect(panel_x - spacing // 2, panel_y - spacing // 2,
                                panel_width + spacing, screen_height - 2 * spacing)
        pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
        pygame.draw.rect(self.screen, border_color, panel_rect, 2)

        # Title for the panel.
        counts_title = font_large.render("Card Counts", True, text_color)
        self.screen.blit(counts_title, (panel_x + (panel_width - counts_title.get_width()) // 2, panel_y))
        
        # Render each card count.
        line_height = font_small.get_height() + 5
        for idx, rank in enumerate(range(1, 11)):
            label = "A" if rank == 1 else str(rank)
            count = card_counts[rank]
            line_text = font_small.render(f"{label}: {count}", True, text_color)
            text_x = panel_x + spacing // 2
            text_y = panel_y + counts_title.get_height() + spacing + idx * line_height
            self.screen.blit(line_text, (text_x, text_y))

        # --- Finalize rendering ---
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(30)  # Limit to 30 FPS
        else:
            return np.array(pygame.surfarray.array3d(self.screen)).transpose(1, 0, 2)


# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)
