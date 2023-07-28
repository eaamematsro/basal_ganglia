import pdb

import numpy as np
import torch

from model_factory.architectures import RNNStaticBG, RNNFeedbackBG, NETWORKS
from model_factory.networks import transfer_network_weights
from model_factory.tasks import GenerateSine


vanilla_model = GenerateSine(network="VanillaRNN")
# vanilla_model.plot_different_gains()
# vanilla_model.training_loop()
cool_model = GenerateSine(network="RNNFeedbackBG")
# transfer_network_weights(cool_model.network.rnn, vanilla_model.network.rnn,
#                          freeze=True)
transfer_network_weights(cool_model.network, vanilla_model.network,
                         freeze=True)
# cool_model.configure_optimizers()
cool_model.training_loop(1)
print("completed")
pdb.set_trace()