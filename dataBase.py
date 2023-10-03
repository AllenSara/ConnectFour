# Imports from Environment and date modules
from imports import *

date = datetime.today()

# Create Connection to MongoDB
g_client = pymongo.MongoClient("mongodb://localhost:27017/")
# Create Database in MongoDB
g_db = g_client["Game_DB_new"]
# Game Data Collection in MongoDB
g_col = g_db["Game_Data"]


# Database class to store game data
class Database:
    def __init__(self):
        self.g_num = []
        self.g_win = []
        self.g_loss = []
        self.g_epsilon = []
        self.g_board = []
        self.clean()

    # Function to clean database after update
    def clean(self):
        self.g_num = []
        self.g_win = []
        self.g_loss = []
        self.g_epsilon = []
        self.g_board = []

    # Update Database Function
    def update_db(self, iteration, win, loss, epsilon, board):
        self.g_num.append(iteration)
        self.g_win.append(win)
        self.g_loss.append(loss)
        self.g_epsilon.append(epsilon)
        self.g_board.append(board)

        g_dict = {
            "Iteration": self.g_num,
            "Winner": self.g_win,
            "Score": self.g_loss,
            "Epsilon": self.g_epsilon,
            "Time": date.ctime(),
            "Board": self.g_board
        }

        # add to database
        g_col.insert_one(g_dict)
