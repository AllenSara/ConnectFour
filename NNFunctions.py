from imports import *

# Define the Q learning neural network model


def q_learning_nn():
    model = keras.Sequential()
    # converts the matrix/cube layer into a one-dimension vector/array which is used as input for a dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))  # 50 - shape of output, relu activation converts input into output
    model.add(layers.Dense(50, activation='relu'))  # relu creates a linear output by changing all negative outputs to 0
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))

    return model

# Functions to train the model


def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss


def train_step(model, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network

        logits = model(observations)
        loss = compute_loss(logits, actions, rewards)
        loss_value = loss.numpy()
        grads = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_value
