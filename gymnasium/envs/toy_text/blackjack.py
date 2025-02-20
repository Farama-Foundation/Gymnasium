import os
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import random

def cmp(a, b):
    return float(a > b) - float(a < b)

"""
  Blackjack is a card game where the goal is to beat the dealer by obtaining cards
  that sum to closer to 21 (without going over 21) than the dealer's cards.

  ## Description
  The game starts with the dealer having one face up and one face down card,
  while the player initially is dealt two cards. All cards are drawn from an infinite deck
  (i.e. with replacement).

  Card values:
  - Face cards (Jack, Queen, King) are worth 10.
  - Aces can count as 11 (usable ace) or 1.
  - Numerical cards (2-10) are worth their number.

  In this modified version, the player's hand is represented by its total sum and
  a flag indicating if there is a usable ace. Splitting is allowed; when the player
  splits, the environment internally stores all resulting hands and plays them sequentially.
  
  ## Action Space
  0: Stick
  1: Hit
  2: Double Down
  3: Split

  ## Observation Space
  The observation is now a 4-tuple:
    (player_total, dealer_showing, usable_ace_flag, true_count)
  where:
    - player_total: the sum of the current (active) hand (Discrete(32), assumed range 0-31)
    - dealer_showing: the dealer’s visible card (Discrete(11), values 0–10)
    - usable_ace_flag: 0 or 1 (Discrete(2))
    - true_count: an integer between 0 and 10 (Discrete(11))
  
  ## Rewards
  - Win: +1 (or +1.5 for a natural blackjack, per your natural flag)
  - Loss: -1 (or -2 when doubling and busting)
  - Draw: 0
  
  For split actions, a legal split returns a reward of 0 immediately and the episode continues.

  ## Episode End
  An episode ends only when all of the player’s hands have been played.

  ## References
  [1] R. Sutton and A. Barto, “Reinforcement Learning: An Introduction” (2020).
"""

class BlackjackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False,
                 num_decks=5, evaluation_mode=False):
        # Actions: 0: Stick, 1: Hit, 2: Double Down, 3: Split.
        self.action_space = spaces.Discrete(4)
        # Observation: (player_total, dealer_showing, usable_ace_flag, true_count)
        # Assume player's total can be between 0 and 31.
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Discrete(11),
            spaces.Discrete(2)
        ))
        self.natural = natural
        self.sab = sab
        self.render_mode = render_mode
        self.num_decks = num_decks
        self.deck = []  # List of available cards
        self.np_random = np.random.default_rng()
        self.running_count = 0
        self.evaluation_mode = evaluation_mode

        # Instead of a single hand, we store all player hands here.
        self.player_hands = []
        self.current_hand_index = 0
        self.dealer = []
        self.dealer_turn = False
        self._reshuffle_deck()
    
    def _reshuffle_deck(self):
        self.deck = self.num_decks * [1,2,3,4,5,6,7,8,9,10,10,10,10] * 4
        self.np_random.shuffle(self.deck)
        self.running_count = 0
    
    def _update_running_count(self, card):
        if card in [2,3,4,5,6]:
            self.running_count += 1
        elif card in [10, 1]:
            self.running_count -= 1
    
    def _get_true_count(self):
        remaining_decks = max(1, len(self.deck) / 52)
        true_count = self.running_count / remaining_decks
        bounded_count = int(round(true_count))
        bounded_count = min(5, max(-5, bounded_count))
        return bounded_count 
    
    def draw_card(self):
        if len(self.deck) == 0:
            self._reshuffle_deck()
        card = self.deck.pop()
        self._update_running_count(card)
        return card
    
    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def sum_hand(self, hand):
        # Compute hand total, counting a usable ace as 11 if it doesn't bust.
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
        # Use the active hand from self.player_hands.
        current_hand = self.player_hands[self.current_hand_index]
        player_total = self.sum_hand(current_hand)
        usable_flag = int(1 in current_hand and sum(current_hand) + 10 <= 21)
        # New flag: hand is splittable if exactly two cards and they are equal.
        splittable_flag = int(len(current_hand) == 2 and current_hand[0] == current_hand[1])
        return (player_total,
                self.dealer[0],
                usable_flag,
                self._get_true_count(),
                splittable_flag)
    
    def step(self, action):
        assert self.action_space.contains(action)
        # We'll operate on the active hand.
        current_hand = self.player_hands[self.current_hand_index]
        
        if action == 1:  # Hit
            current_hand.append(self.draw_card())
            if self.is_bust(current_hand):
                reward = -1.0
                # If there are more hands left, move to next hand.
                if self.current_hand_index < len(self.player_hands) - 1:
                    self.current_hand_index += 1
                    return self._get_obs(), reward, False, False, {}
                else:
                    return self._get_obs(), reward, True, False, {}
            else:
                return self._get_obs(), 0.0, False, False, {}
        
        elif action == 0:  # Stick
            # Player sticks on the active hand.
            self.dealer_turn = True  # Set dealer's turn to True
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = cmp(self.score(current_hand), self.score(self.dealer))
            # Adjust reward for naturals if applicable.
            if self.sab and self.is_natural(current_hand) and not self.is_natural(self.dealer):
                reward = 1.0
            elif not self.sab and self.natural and self.is_natural(current_hand) and reward == 1.0:
                reward = 1.5
            # Move on to next hand if available.
            if self.current_hand_index < len(self.player_hands) - 1:
                self.current_hand_index += 1
                return self._get_obs(), reward, False, False, {}
            else:
                return self._get_obs(), reward, True, False, {}
        
        elif action == 2:  # Double Down
            current_hand.append(self.draw_card())
            if self.is_bust(current_hand):
                reward = -2.0
                if self.current_hand_index < len(self.player_hands) - 1:
                    self.current_hand_index += 1
                    return self._get_obs(), reward, False, False, {}
                else:
                    return self._get_obs(), reward, True, False, {}
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = 2 * cmp(self.score(current_hand), self.score(self.dealer))
            if self.current_hand_index < len(self.player_hands) - 1:
                self.current_hand_index += 1
                return self._get_obs(), reward, False, False, {}
            else:
                return self._get_obs(), reward, True, False, {}
        
        elif action == 3:
            # Split: valid only if active hand has exactly 2 cards and they are equal.
            if len(current_hand) == 2 and current_hand[0] == current_hand[1]:
                # Remove one card and form a new hand.
                split_card = current_hand.pop()
                new_hand = [split_card]
                # Deal one new card for each hand.
                current_hand.append(self.draw_card())
                new_hand.append(self.draw_card())
                # Append the new hand.
                self.player_hands.append(new_hand)
                # Return observation with reward 0 (the split itself gives no immediate reward).
                return self._get_obs(), 0.0, False, False, {}
            else:
                # Illegal split: penalty and end the episode.
                return self._get_obs(), -20.0, True, False, {}

    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.evaluation_mode:
            self._reshuffle_deck()
            max_discards = len(self.deck) - 15
            num_discards = self.np_random.integers(low=0, high=max_discards + 1)
            for _ in range(num_discards):
                self.deck.pop()
        else:
            if len(self.deck) < 15:
                self._reshuffle_deck()
        # Reset dealer and player hands.
        self.dealer = self.draw_hand()
        self.player_hands = [self.draw_hand()]
        self.current_hand_index = 0
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

        bg_color = (0, 128, 0)
        self.screen.fill(bg_color)

        # Correct the path to the image folder based on the current script location
        card_image_folder = os.path.join(os.path.dirname(__file__), "img")  # Absolute path to './img'

        import random

        def get_card_suit_and_rank(card_value):
            """Map card values to a tuple (suit, rank)."""
            suits = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
            ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
            
            # If the card value is 10, pick randomly between T, J, Q, and K
            if card_value == 10:
                rank = random.choice(['T', 'J', 'Q', 'K'])
            else:
                rank = ranks[(card_value - 1) % 13]  # Adjust for 0-based indexing

            # Randomly pick a suit
            suit = suits[random.randint(0, 3)]  # Map to suit based on index
            
            return suit, rank


        def draw_card(surface, suit, rank, x, y, width, height):
            """Draw the card using the corresponding PNG image."""
            filename = f"{suit}{rank}.png"  # Construct the filename (XY.png format)
            image_path = os.path.join(card_image_folder, filename)
            try:
                image = pygame.image.load(image_path)
                image = pygame.transform.scale(image, (width, height))  # Resize to fit the card size
                surface.blit(image, (x, y))  # Draw the image on the surface
            except pygame.error:
                print(f"Warning: {filename} not found. Drawing a blank card.")
                # If the specific card image is not found, use a generic card back
                back_image_path = os.path.join(card_image_folder, "Card.png")
                back_image = pygame.image.load(back_image_path)
                back_image = pygame.transform.scale(back_image, (width, height))
                surface.blit(back_image, (x, y))  # Draw the back image on the surface
        
        # --- Render Dealer's Hand ---
        dealer_title = pygame.font.SysFont("Arial", 32).render("Dealer's Hand", True, (255, 255, 255))
        self.screen.blit(dealer_title, (10, 10))
        dealer_x = 10
        dealer_y = 50
        for i, card in enumerate(self.dealer):
            suit, rank = get_card_suit_and_rank(card)  # Get the suit and rank for the card
            x = dealer_x # Adjust card spacing
            if not self.dealer_turn and i == 1:  # Hide the second card until dealer's turn
                draw_card(self.screen, 'Card', '', x, dealer_y, 60, 90)  # Draw the back of the card
            else:
                draw_card(self.screen, suit, rank, x, dealer_y, 60, 90)
            dealer_x += 70

        # Calculate dealer's hand total
        dealer_total = self.sum_hand(self.dealer)
        dealer_bust = dealer_total > 21

        # Render Dealer's Total and Bust Status
        dealer_total_text = pygame.font.SysFont("Arial", 24).render(f"Total: {dealer_total}", True, (255, 255, 255))
        self.screen.blit(dealer_total_text, (dealer_x + 10, dealer_y + 20))  # Position the total
        if dealer_bust:
            dealer_bust_text = pygame.font.SysFont("Arial", 24).render("Busted!", True, (255, 0, 0))
            self.screen.blit(dealer_bust_text, (dealer_x + 10 , dealer_y + + 50))  # Position the bust message

        # --- Render Player's Hands ---
        player_title = pygame.font.SysFont("Arial", 32).render("Player's Hands", True, (255, 255, 255))
        self.screen.blit(player_title, (10, dealer_y + 100))
        player_y = dealer_y + 150

        # Right side: Render True Count
        true_count_title = pygame.font.SysFont("Arial", 32).render(f"True Count: {self._get_true_count()}", True, (255, 255, 255))
        self.screen.blit(true_count_title, (screen_width - 300, 10))

        # Render Running Count below the True Count
        running_count_title = pygame.font.SysFont("Arial", 32).render(f"Running Count: {self.running_count}", True, (255, 255, 255))
        self.screen.blit(running_count_title, (screen_width - 300, 40))  # Adjust position below True Count

        # Render player hands with totals
        for hand in self.player_hands:
            hand_x = 10
            hand_total = self.sum_hand(hand)
            bust = hand_total > 21
            # Render hand cards
            for card in hand:
                suit, rank = get_card_suit_and_rank(card)  # Get the suit and rank for the card
                draw_card(self.screen, suit, rank, hand_x, player_y, 60, 90)
                hand_x += 70
            
            # Render player total and bust message
            total_text = pygame.font.SysFont("Arial", 24).render(f"Total: {hand_total}", True, (255, 255, 255))
            self.screen.blit(total_text, (hand_x + 10, player_y + 20))  # Adjust position of the total
            if bust:
                bust_text = pygame.font.SysFont("Arial", 24).render("Busted!", True, (255, 0, 0))
                self.screen.blit(bust_text, (hand_x + 10, player_y + 50))

            player_y += 120  # Space between player hands

        # Display the screen
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(30)
