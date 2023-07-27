import os
import torch
import re
import glob
import pdb
import pickle
import itertools
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from datetime import date
from scipy.stats import special_ortho_group
from model_factory.networks import transfer_network_weights
from model_factory.tasks import GenerateSine

plt.style.use('ggplot')

if __name__ == '__main__':
    simple_model = GenerateSine(network="VanillaRNN")
    simple_model.training_loop(niterations=1000)

    thalamic_model = GenerateSine(network="RNNStaticBG")

    # Transfer and freeze weights from trained network's rnn module
    transfer_network_weights(thalamic_model.network.rnn, simple_model.network.rnn,
                             freeze=True)

    # Change target parameters to learn
    thalamic_model.create_gos_and_targets(frequency=2)
    thalamic_model.training_loop(niterations=5000)


