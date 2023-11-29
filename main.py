from memory import *
from NNFunctions import *
from dataBase import *
from ConnectEnv import *

# Create a new instance of and initialize classes
game = ConnectEnv()
memory = Memory()
Database = Database()

# Time collection variables
now = datetime.now()
hour = now.hour
minute = now.minute
second = now.second
date = date.today()

# Define the size of the state and action spaces
state_size = game.ROW_COUNT * game.COLUMN_COUNT  # 6 rows x 7 columns
action_size = game.COLUMN_COUNT

# Set the number of episodes to train for
num_episodes = 15

# Define the learning rate and optimizer
tf.keras.backend.set_floatx('float64')
LEARNING_RATE = 0.0001
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# runs the model, either from the existing model or by creating a new one
if Path('connect_four_model.h5').is_file():
    model = keras.models.load_model('connect_four_model.h5')
    print("used model")
else:
    model = q_learning_nn()
    print("new model")

# initializes the epsilon, either from the existing epsilon or by creating a new one
if Path('epsilon.txt').is_file():
    with open('epsilon.txt', 'r') as f:
        epsilon = f.read()
    print("used epsilon")
else:
    epsilon = 1
    print("new epsilon")

# try to run model for set amount of episodes
try:

    # Loop through each episode
    for episode in range(num_episodes):

        # Reset the game variables to their starting state
        game.reset()

        # Set the initial/blank state for the board
        state = game.create_board()

        # Set for first time/decrease the epsilon slightly for each game
        epsilon = float(epsilon) * 0.999985

        print(f'Game Number: {episode + 1}')

        # Loop which runs for each individual game until the game ends
        while not game.game_over == True:

            # Turn for the player AKA the model being trained
            if game.turn == game.PLAYER:
                # reshape state for prediction model, so it can return an action option between 0-6
                reshaped_state = np.array(state).reshape(1, 6, 7, 1).copy()

                # Create the Q values using the reshaped state
                # the Q value is a measure of the overall expected reward based on the action
                q_values = model.predict(reshaped_state)

                # Convert q_values into positive numbers
                action_weights = tf.nn.softmax(q_values).numpy()

                # Creates array of tuples
                # Each tuple contains the column/action (index of q_value) and weight (q_value) corresponding to each action
                # Sorts the tuples from highest to lowest
                action_weights_sorted = sorted(enumerate(action_weights[0]), key=lambda i: i[1], reverse=True)

                # Find a valid action: Loop through action options from the q_values
                # Because they're sorted from highest to lowest, starts at first one and uses first valid action option
                for action_test in enumerate(action_weights_sorted):

                    # Check if the action is valid, if not the for loop will move to the next highest weight
                    # The use of action[1][0] extracts the column for the action from the action tuple
                    if action_test[1][0] in game.get_valid_locations(state):

                        # Use valid column to drop the piece (step), and save the variables
                        reward, next_state, action = game.step(action_test[1][0], epsilon)

                        # check if valid action was taken by seeing if the turn was switched to AI (the end of the step)
                        if game.turn == game.AI:

                            # add the newest state (board), action taken, and calculate reward to the memory
                            # this will be used at the end of the game to calculate the reward
                            memory.add_to_memory(state, action, reward)

                        # break once a valid action is taken and then run the next move or end the game
                        break

            # run the AIs turn AKA the algorithm that uses calculated moves
            elif game.turn == game.AI:

                # Run the game and save the variables, the reward and action will not be used
                reward, next_state, action = game.step()

            # Set the current state (board) to the new state (board) with the newly dropped piece
            state = next_state

        # Train the model by inputting the memory of the game in the train_step function
        loss = train_step(model, optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                          observations=np.array(memory.observations), actions=np.array(memory.actions),
                          rewards=memory.rewards)

        # shape and then upload data to mongoDB database
        state = state.tolist()
        Database.update_db(episode, game.winner, loss, epsilon, state)

        # Save the trained model to an h5 file after every 5000th game - iteration, date and time
        # these models are backups. They use the iteration and time to have a unique value and will not overwrite files
        if (episode+1) % 5000 == 0:
            model.save(f'Iteration_{episode+1}_Date-{date}-Time-{hour}{minute}{second}.h5')
            print(f'Saved model number {episode}')

        # clear the memory and database variables, so they can be used for the next game
        memory.clear()
        Database.clean()

# after all episodes are finished, save the model to be used for the next set of runs or just to play against
# this model has the same name and will thus overwrite the initial model used
finally:
    model.save(f'connect_four_model.h5')
    print(f'Final episode number {episode + 1}')

# save the epsilon to a txt file which can be used for the next set of runs
with open('epsilon.txt', 'w') as f:
    f.write(str(epsilon))
    print(f'Epsilon: {epsilon}')
