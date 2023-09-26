import pdb
import matplotlib.pyplot as plt
from model_factory.networks import transfer_network_weights
from datasets.tasks import GenerateSine
from datasets.loaders import SineDataset
from torch.utils.data import DataLoader, RandomSampler

plt.style.use("ggplot")

if __name__ == "__main__":
    simple_model = GenerateSine(network="VanillaRNN")
    dataset = SineDataset(n_unique_pulses=50)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size=12, sampler=sampler)

    simple_model.training_loop(data_loader=loader, niterations=1000)
    simple_model.save_model()
    thalamic_model = GenerateSine(network="RNNStaticBG", nbg=50)

    # Transfer and freeze weights from trained network's rnn module
    transfer_network_weights(thalamic_model.network, simple_model.network, freeze=True)
    # thalamic_model.network.rnn.reconfigure_u_v()

    # Change target parameters to learn
    thalamic_model.create_gos_and_targets(frequency=2)
    thalamic_model.training_loop(
        data_loader=train_dataloaders, niterations=1000, batch_size=24
    )
    thalamic_model.save_model()

    # Interpolate over gains
    thalamic_model.plot_different_gains()
    pdb.set_trace()
