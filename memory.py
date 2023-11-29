# This class stores game data for each game and is used to train the model

class Memory:
    # initialize variables
    def __init__(self):
        self.track_scores = []
        self.rewards = []
        self.actions = []
        self.observations = []
        self.clear()

    # Reset the memory to blank state
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Adds game state (board), action, and reward to the corresponding lists
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(float(new_reward))

    # Add the game iteration, who won the game (0 is PLAYER and 1 is RND), the winning move, and the loss score
    # def add_game_score(self, iteration, win, move, loss):
    #    self.track_scores.append((iteration, win, move, loss))
