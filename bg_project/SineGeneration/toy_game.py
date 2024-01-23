import pdb

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, gain: float = 1):
    return 1.0 / (1.0 + np.exp(-x / gain))


n_feature = 5
n_cards = 5

features = np.eye(n_feature)

cards = np.random.randn(n_feature, n_cards)

W_sun = np.random.randn(n_feature, 1)
W_rain = np.random.randn(n_feature, 1)

p_sun = sigmoid(W_sun.T @ features)
p_rain = sigmoid(W_rain.T @ features)
l_sun = p_sun / (p_rain + p_sun)

trials = 15000
card_number = 0
guesses = 0
p_net = np.ones(n_cards) * 0.5
errors = []
lr = 1e-3
gamma = 1
gammas = np.logspace(
    -1,
    2,
)

for trial in range(trials):
    alpha_exp = 10 * np.exp(-trial / 100) + 0.5
    card_number = np.random.randint(0, n_cards)  # Choose a card from the set of cards
    card = cards[:, card_number]
    # true_prob = l_sun[0, card_number]
    true_prob = l_sun[0, card_number]
    actual_outcome = np.random.choice([0, 1], 1, p=[1 - true_prob, true_prob])

    probs = np.exp(np.asarray([1 - p_net[card_number], p_net[card_number]]) / alpha_exp)
    probs /= probs.sum()
    try:
        network_output = np.random.choice([0, 1], p=probs)
    except:
        pdb.set_trace()

    error = actual_outcome - network_output
    p_net[card_number] = p_net[card_number] + lr * error[0]
    # guess = np.random.choice([0, 1], 1)
    guesses += actual_outcome
    errors.append(error)


plt.scatter(l_sun, p_net)
plt.xlabel("True Prob")
plt.ylabel("Learnt prob")
plt.pause(0.1)
pdb.set_trace()
