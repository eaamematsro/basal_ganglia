import numpy as np
import torch

from architectures import RNNStaticBG, RNNFeedbackBG, NETWORKS
from tasks import GenerateSine
from factory_utils import torchify

test_model = GenerateSine(network="VanillaRNN")
test_model.training_loop()