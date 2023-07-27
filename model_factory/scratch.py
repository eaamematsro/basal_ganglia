import numpy as np
import torch

from architectures import RNNStaticBG, RNNFeedbackBG, NETWORKS
from networks import transfer_network_weights
from tasks import GenerateSine


vanilla_model = GenerateSine(network="VanillaRNN")
# vanilla_model.training_loop()
cool_model = GenerateSine()
transfer_network_weights(cool_model.network.rnn, vanilla_model.network.rnn,
                         freeze=True)
cool_model.configure_optimizers()