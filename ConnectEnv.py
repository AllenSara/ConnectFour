from imports import *


class ConnectEnv(Env):
    metadata = {'render.modes': ['human']}

    # the code below up until the init function is adapted from Keith's code

    # set variables for size of columns and rows to use throughout the game
    ROW_COUNT = 6
    COLUMN_COUNT = 7

    # set amount in a row needed to win
    WINDOW_LENGTH = 4

    # set player variables (used for turns)
    PLAYER = 0
    AI = 1

    # set piece variables (used to fill board)
    EMPTY = 0
    PLAYER_PIECE = 1
    AI_PIECE = 2

    # create board by making a matrix of all zeros using the predefined row and column sizes
    def create_board(self):
        board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT)).astype(int)  # uses numpy
        return board

    # fills in the board with the dropped piece
    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece

    # checks if the top spot of the column selected is empty, returns true or false
    def is_valid_location(self, board, col):
        return board[self.ROW_COUNT - 1][col] == 0  # ROW_COUNT - 1 is the fifth or top row bc index starts at 0

    # checks where the piece will fall within the column, returns row to drop piece into
    # loops through all the rows of the column, starts at the bottom and returns the first empty spot
    def get_next_open_row(self, board, col):
        for r in range(self.ROW_COUNT):
            if board[r][col] == 0:
                return r

    # prints a flipped version of the board
    # flips the board so the zero axis is on the bottom of the game
    def print_board(self, board):
        print(np.flip(board, 0))

    # checks if someone won the game by checking whole board for four in a row
    def winning_move(self, board, piece):
        # Check all horizontal locations for win:
        # loop through each column, meaning starting at the left or zero index and iterate to the right
        # last three columns can't have a horizontal win bc there aren't four spots left
        # in each column, loop through each row, meaning from the bottom or zero index and then iterate up to the top
        # iterate through the row manually (c+1 each time) to check if the four spots equal the player's piece
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT):
                if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                    c + 3] == piece:
                    return True

        # Check all vertical locations for win:
        # loop through each column
        # in each column, loop through each row
        # last three rows can't have a vertical win bc there aren't four spots left
        # iterate up the column manually (r+1 each time) to check if the four spots equal the player's piece
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT - 3):
                if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                    c] == piece:
                    return True

        # Check all positively sloped diagonals:
        # loop through each column, last three can't have a diagonal win
        # in each column, loop through each row, last three can't have a diagonal win
        # iterate both up (r+1) and to the right (c+1) to check for four in a row
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT - 3):
                if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and \
                        board[r + 3][c + 3] == piece:
                    return True

        # Check negatively sloped diagonals:
        # loop through each column, last three can't have a diagonal win
        # in each column, loop through each row, first four can't have a diagonal win
        # iterate both up (r+1) and to the right (c+1) to check for four in a row
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(3, self.ROW_COUNT):  # start at 4th row which is the third index
                if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and \
                        board[r - 3][c + 3] == piece:
                    return True

    # check is there's a stalemate
    def board_full(self, board):
        if not np.any(board == 0):
            return True
        else:
            return False

    # returns an array of all the valid locations, AI player uses this so its moves are always valid
    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(self.COLUMN_COUNT):
            if self.is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations

    # calculates a score based on winning potential of an inputted window of four spaces
    # used for both the model and the min max
    def evaluate_window(self, window, piece):
        score = 0  # initialize score

        # set which player to calculate the score for
        model = False
        opp_piece = self.PLAYER_PIECE
        if piece == self.PLAYER_PIECE:
            opp_piece = self.AI_PIECE
            model = True

        if window.count(piece) == 4:  # add score if there are four of the players pieces aka a win
            score += 10
        if window.count(piece) == 3 and window.count(
                self.EMPTY) == 1:  # add score if there are 3 of the players pieces and an empty piece
            score += 0.5
        elif window.count(piece) == 2 and window.count(
                self.EMPTY) == 2:  # add score if there are 2 of the players pieces and 2 empty pieces
            score += 0.2
        if model:
            if window.count(opp_piece) == 3 and window.count(
                    piece) == 1:  # add score if there are 3 of the opponents pieces and the player blocked the 4 spot
                score += 3
            # add score if player blocked two in a row from opponent
            # this method sometimes calculates points not reflected on the players move
            # for this reason the added score is very low
            elif window.count(opp_piece) == 2 and window.count(piece) == 1 and window.count(self.EMPTY) == 1:
                score += 0.05
        else:
            if window.count(opp_piece) == 3 and window.count(
                    self.EMPTY) == 1:  # subtract score if the opponent has three in a row and an empty piece
                score -= 40
        return score

    # breaks up entire board into all possible windows of four / possible places where there could be a win
    # and runs the evaluate window function of each window of four in the board, totals the score for everything
    def score_position(self, board, piece):
        score = 0  # initialize score
        # Score center column
        center_array = [int(i) for i in list(board[:, self.COLUMN_COUNT // 2])]
        center_count = center_array.count(piece)
        score += center_count * 1

        # Score Horizontal
        for r in range(self.ROW_COUNT):  # for each row
            row_array = [int(i) for i in list(board[r, :])]  # for this specific row check each column in the row
            for c in range(self.COLUMN_COUNT - 3):  # look at each window size of four, -3 bc those aren't scorable
                window = row_array[c:c + self.WINDOW_LENGTH]  # get four at a time and check what pieces are there
                score += self.evaluate_window(window, piece)

        # Score Vertical
        for c in range(self.COLUMN_COUNT):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(self.ROW_COUNT - 3):
                window = col_array[r:r + self.WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score positive sloped diagonal
        for r in range(self.ROW_COUNT - 3):
            for c in range(self.COLUMN_COUNT - 3):
                window = [board[r + i][c + i] for i in range(self.WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        # Score negative sloped diagonal
        for r in range(self.ROW_COUNT - 3):
            for c in range(self.COLUMN_COUNT - 3):
                window = [board[r + 3 - i][c + i] for i in range(self.WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score

    # min max uses this to choose its next move
    def pick_best_move(self, board):
        valid_locations = self.get_valid_locations(board)
        best_score = -10000
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = self.get_next_open_row(board, col)
            temp_board = board.copy()
            self.drop_piece(temp_board, row, col, self.AI_PIECE)
            score = self.score_position(temp_board, self.AI_PIECE)
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

    # model uses this to receive punishments for any place where it hasn't blocked the opponents set of three
    def calculate_punishment(self, board):
        valid_locations = self.get_valid_locations(board)
        punishment = 0
        for col in valid_locations:
            temp_board = board.copy()
            row = self.get_next_open_row(temp_board, col)
            self.drop_piece(temp_board, row, col, self.AI_PIECE)
            if self.winning_move(temp_board, self.AI_PIECE):
                punishment -= 3
        return punishment

    # The initial starting function, builds the whole environment when you first start it
    def __init__(self):
        # I think this command runs the init function when the environment is created
        super(ConnectEnv, self).__init__()
        self.observation_space = self.create_board()
        # Actions user can take, i.e. choosing a column from 1-7
        self.action_space = Discrete(7)
        # variable we can check to see if game ended
        self.game_over = False
        # variable to switch between turns, starts off randomly
        self.turn = random.randint(self.PLAYER, self.AI)
        # variables to count wins for each player
        self.winner = -1
        self.player_wins = 0
        self.ai_wins = 0
        self.reward = 0

    # Does the actions, basically checks which player is playing and then drops a piece for them accordingly
    def step(self, action=1, epsilon=0.999985):
        if self.turn == self.AI:
            col = self.pick_best_move(self.observation_space)
            if self.is_valid_location(self.observation_space, col):
                # finds the row the piece will be dropped in for the chosen column
                row = self.get_next_open_row(self.observation_space, col)
                self.drop_piece(self.observation_space, row, col, self.AI_PIECE)

                if self.winning_move(self.observation_space, self.AI_PIECE):
                    self.game_over = True
                    self.winner = self.AI
                    # print("AI won the game!")
                    self.ai_wins += 1
                    # self.reward = -100

                self.turn += 1
                self.turn = self.turn % 2

        elif self.turn == self.PLAYER:
            valid_locations = self.get_valid_locations(self.observation_space)
            rnd = random.choice(valid_locations)
            # the column is chosen as either the models choice or a random valid location
            # the epsilon determines how often the choice will be random vs the model
            col = np.random.choice([action, rnd], 1, p=[1 - epsilon, epsilon])[0]
            if self.is_valid_location(self.observation_space, col):
                # the reward function is calculated based on the state of the entire board, and not just the piece most
                # recently dropped. To give a reward on the specific action taken we find the difference between
                # the score before and after taking the action
                score_before = self.score_position(self.observation_space, self.PLAYER_PIECE)
                # finds the row the piece will be dropped in for the chosen column
                row = self.get_next_open_row(self.observation_space, col)
                self.drop_piece(self.observation_space, row, col, self.PLAYER_PIECE)
                score_after = self.score_position(self.observation_space, self.PLAYER_PIECE)
                punishment = self.calculate_punishment(self.observation_space)
                # print(f'before: {score_before} after: {score_after} punishment: {punishment}')
                self.reward = score_after - score_before + punishment
                if self.winning_move(self.observation_space, self.PLAYER_PIECE):
                    self.game_over = True
                    self.winner = self.PLAYER
                    # print("PLAYER won the game!")
                    self.player_wins += 1
                    # bc we want winning to be chosen above all else, the reward is reset here
                    self.reward = 100

                self.turn += 1
                self.turn = self.turn % 2

        # bc the board prints at the end, "__ won the game!" will print before the winning board
        # self.print_board(self.observation_space)

        # check if board is full AKA a stalemate, and if it is end the game
        if self.board_full(self.observation_space):
            self.game_over = True

            # print("Stalemate")

        return self.reward, self.observation_space, col

    # resets the game (but not the whole environment) so the environment can run a new game while keeping old data
    def reset(self):
        self.observation_space = self.create_board()
        self.turn = random.randint(self.PLAYER, self.AI)
        self.game_over = False
