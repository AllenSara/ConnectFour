from memory import Memory
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

if Path('epsilon.txt').is_file():
    with open('epsilon.txt', 'r') as f:
        epsilon = f.read()
    print("used epsilon")
else:
    epsilon = 1
    print("new epsilon")

try:

    # Loop through each episode
    for episode in range(num_episodes):

        # Reset the game
        game.reset()

        # Get the initial/blank state
        state = game.create_board()

        # Calls the clear method on the memory object, which resets/restarts the memory

        epsilon = float(epsilon) * 0.999985
        # print(epsilon)

        print(f'Game Number: {episode + 1}')
        # Loop for each individual game
        while not game.game_over == True:
            if game.turn == game.PLAYER:
                # Change shape of state so the prediction model can understand it and return the right action options (0-6)
                reshaped_state = np.array(state).reshape(1, 6, 7, 1).copy()

                # Make a prediction using the Q values and the reshaped state
                # the Q value is a measure of the overall expected reward based on the action
                q_values = model.predict(reshaped_state)

                # Convert q_values into positive numbers
                action_weights = tf.nn.softmax(q_values).numpy()

                # Creates array of tuples
                # Each tuple contains the column/action (index of q_value) and weight (q_value) corresponding to each action
                # Sorts the tuples from highest to lowest
                action_weights_sorted = sorted(enumerate(action_weights[0]), key=lambda i: i[1], reverse=True)

                # Find a valid action: Loop through action options from the q_values
                for action_test in enumerate(action_weights_sorted):

                    # Check if the action is valid, if not the for loop will move to the next highest weight
                    # The use of action[1][0] extracts the column for the action from the action tuple
                    if action_test[1][0] in game.get_valid_locations(state):

                        # Now that we have a valid column, do the action (step function) and save the variables
                        reward, next_state, action = game.step(action_test[1][0], epsilon)
                        # print(f'action: {action}')
                        # print(f'reward: {reward}')



                        # Set the current state to the next state
                        state = next_state

                        # Adds the current observation, action, and reward to the memory
                        # bc the turn switches when we run the step, check if action was taken and then save it
                        if game.turn == game.AI:
                            memory.add_to_memory(state, action, reward)

                        # break once a valid action is taken
                        break

            elif game.turn == game.AI:
                # We need to input an action into the step, but this action (1) will not be used
                reward, next_state, action = game.step(1, epsilon)
                state = next_state

                # There's no else here for the case where the entire board is full
                # it should be caught in the while loop above in "game_over", but maybe add an extra check here
        # print(memory.observations)
        # print(memory.actions)
        # print(memory.rewards)
        # Train the model using the train_step function
        loss = train_step(model, optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                          observations=np.array(memory.observations), actions=np.array(memory.actions),
                          rewards=memory.rewards)

        # Upload data to database
        state = state.tolist()
        Database.update_db(episode, game.winner, loss, epsilon, state)
        if (episode+1) % 5000 == 0:

            # Save trained model after session - iteration, date and time.
            model.save(f'Iteration_{episode}_Date-{date}-Time-{hour}{minute}{second}.h5')
            print(f'Saved model number {episode}')

        memory.clear()
        Database.clean()
finally:

    model.save(f'connect_four_model.h5')
    print(f'Final episode number {episode + 1}')


with open('epsilon.txt', 'w') as f:
    f.write(str(epsilon))
    print(f'Epsilon: {epsilon}')

