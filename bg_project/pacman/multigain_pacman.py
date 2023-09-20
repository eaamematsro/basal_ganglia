import pdb
import torch
import matplotlib.pyplot as plt
from typing import Sequence
from model_factory.networks import transfer_network_weights
from datasets.tasks import MultiGainPacMan, set_results_path
from datasets.loaders import PacmanDataset
from torch.utils.data import DataLoader, RandomSampler, random_split
from pytorch_lightning import Trainer
plt.style.use('ggplot')


def split_dataset(dataset, fractions: Sequence = (.6, .2, .2)):
    train_set, val_set, test_set = random_split(dataset, fractions)
    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(val_set)
    test_sampler = RandomSampler(test_set)
    train = {'data': train_set, 'sampler': train_sampler}
    val = {'data': val_set, 'sampler': val_sampler}
    test = {'data': test_set, 'sampler': test_sampler}
    return train, val, test


if __name__ == '__main__':
    test_networks = ["RNNMultiContextInput", "RNNStaticBG",]

    torch.set_float32_matmul_precision('medium')
    simple_model = MultiGainPacMan(network="VanillaRNN", bg_input_size=3)

    dataset = PacmanDataset(n_samples=500, masses=(1,), viscosity=(0,), polarity=(1,))
    batch_size = 10

    train_set, val_set, test_set = split_dataset(dataset, (.6, .2, .2))
    # val_contexts = []
    # test_contexts = []
    # plt.figure()
    # plt.title('Validation')
    # for context, trajectory in val_set['data']:
    #     val_contexts.append(context)
    #     plt.plot(trajectory)
    # plt.pause(0.1)
    # plt.figure()
    # plt.title('Testing')
    # for context, trajectory in test_set['data']:
    #     test_contexts.append(context)
    #     plt.plot(trajectory)
    # plt.pause(0.1)
    # pdb.set_trace()

    train_loader = DataLoader(train_set['data'], batch_size=batch_size, sampler=train_set['sampler'], num_workers=10)
    val_loader = DataLoader(val_set['data'], batch_size=batch_size, num_workers=10)

    save_path = set_results_path(type(simple_model).__name__)[0]

    trainer = Trainer(max_epochs=50, gradient_clip_val=1,
                      accelerator='gpu', devices=4, default_root_dir=save_path,
                      )
    trainer.fit(model=simple_model, train_dataloaders=train_loader, val_dataloaders=val_loader
                )

    trainer.test(simple_model, dataloaders=DataLoader(test_set['data'], num_workers=10))

    simple_model.save_model()
    for network in test_networks:
        for nbg in [10, 25, 50]:
            thalamic_model = MultiGainPacMan(network=network, nbg=nbg, bg_input_size=3)

            # Transfer and freeze weights from trained network's rnn module
            transfer_network_weights(thalamic_model.network, simple_model.network,
                                     freeze=True)
            # thalamic_model.network.rnn.reconfigure_u_v()

            train_set, val_set, test_set = split_dataset(PacmanDataset(n_samples=25), (.6, .2, .2))

            train_loader = DataLoader(train_set['data'], batch_size=batch_size,
                                      sampler=train_set['sampler'], num_workers=10)
            
            val_loader = DataLoader(val_set['data'], batch_size=batch_size, num_workers=10)

            save_path = set_results_path(type(thalamic_model).__name__)[0]

            trainer = Trainer(max_epochs=500, gradient_clip_val=1,
                              accelerator='gpu', devices=4, default_root_dir=save_path,
                              )
            trainer.fit(model=thalamic_model, train_dataloaders=train_loader, val_dataloaders=val_loader
                        )

            trainer.test(thalamic_model, dataloaders=DataLoader(test_set, num_workers=10))
            thalamic_model.save_model()





