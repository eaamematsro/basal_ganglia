import pdb
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Sequence
from model_factory.networks import transfer_network_weights
from datasets.tasks import MultiGainPacMan, set_results_path
from datasets.loaders import PacmanDataset
from torch.utils.data import DataLoader, RandomSampler, random_split
from pytorch_lightning import Trainer
plt.style.use("ggplot")


def split_dataset(dataset, fractions: Sequence = (0.6, 0.2, 0.2)):
    train_set, val_set, test_set = random_split(dataset, fractions)
    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(val_set)
    test_sampler = RandomSampler(test_set)
    train = {"data": train_set, "sampler": train_sampler}
    val = {"data": val_set, "sampler": val_sampler}
    test = {"data": test_set, "sampler": test_sampler}
    return train, val, test


if __name__ == "__main__":
    test_networks = ["RNNMultiContextInput", "RNNStaticBG", ]

    weight_penalties = np.logspace(-5, -1)
    torch.set_float32_matmul_precision("medium")
    trial_duration = 150

    for weight_penalty in weight_penalties:

        simple_model = MultiGainPacMan(network="VanillaRNN", duration=trial_duration, apply_energy_penalty=("r_act",),
                                       output_weight_penalty=weight_penalty, bg_input_size=3)
        dataset = PacmanDataset(
            n_samples=500,
            masses=(1,),
            viscosity=(0,),
            polarity=(1,),
            trial_duration=150,
        )
        batch_size = 25

        train_set, val_set, test_set = split_dataset(dataset, (0.6, 0.2, 0.2))

        train_loader = DataLoader(
            train_set["data"],
            batch_size=batch_size,
            sampler=train_set["sampler"],
            num_workers=10,
        )
        val_loader = DataLoader(val_set["data"], batch_size=batch_size, num_workers=10)

        save_path = set_results_path(type(simple_model).__name__)[0]

        # trainer = Trainer(
        #     max_epochs=300,
        #     gradient_clip_val=10,
        #     accelerator="gpu",
        #     devices=1,
        #     default_root_dir=save_path,
        # )
        #
        #
        # trainer.fit(
        #     model=simple_model,
        #     train_dataloaders=train_loader,
        #     val_dataloaders=val_loader,
        # )


        # for batch_idx, batch in enumerate(val_loader):
        #     simple_model.evaluate_training(batch)
        #
        # pdb.set_trace()
        # plt.close('all')
        #
        # for batch_idx, batch in enumerate(val_loader):
        #     simple_model.change_context(batch, new_context=(2, 0, 1))
        #
        # pdb.set_trace()

        # trainer.test(
        #     simple_model, dataloaders=DataLoader(test_set["data"], num_workers=10)
        # )

        # simple_model.save_model()
        file_path = "/home/elom/Documents/basal_ganglia/data/models/2023-09-29/model_1/model.pickle"
        with open(file_path, 'rb') as h:
            loaded_data = pickle.load(h)
        trained_network = loaded_data['network']
        simple_model.network = trained_network

        train_set, val_set, test_set = split_dataset(
            PacmanDataset(n_samples=25, trial_duration=trial_duration,
                          polarity=(-1, 1)),
            (0.6, 0.2, 0.2),
        )

        train_loader = DataLoader(
            train_set["data"],
            batch_size=batch_size,
            sampler=train_set["sampler"],
            num_workers=10,
        )

        val_loader = DataLoader(
            val_set["data"], batch_size=batch_size, num_workers=10
        )

        for network in test_networks:
            for nbg in [10, 25, 50, 100]:
                thalamic_model = MultiGainPacMan(
                    network=network, duration=trial_duration, nbg=nbg,
                    apply_energy_penalty=("r_act", "bg_act"), output_weight_penalty=0,
                    bg_input_size=3, teacher_output_penalty=weight_penalty)

                # Transfer and freeze weights from trained network's rnn module
                transfer_network_weights(
                    thalamic_model.network, simple_model.network, freeze=True
                )

                if network == 'RNNStaticBG':
                    thalamic_model.network.rnn.reconfigure_u_v(g1=1)

                # Training on an easier condition set to get better initializations

                # train_set, val_set, test_set = split_dataset(
                #     PacmanDataset(
                #         n_samples=25, trial_duration=trial_duration, polarity=(1,)
                #     ),
                #     (0.6, 0.2, 0.2),
                # )
                #
                # burn_train_loader = DataLoader(
                #     train_set["data"],
                #     batch_size=batch_size,
                #     sampler=train_set["sampler"],
                #     num_workers=10,
                # )
                #
                # val_loader = DataLoader(
                #     val_set["data"], batch_size=batch_size, num_workers=10
                # )
                #
                # save_path = set_results_path(type(thalamic_model).__name__)[0]
                #
                # burn_trainer = Trainer(
                #     max_epochs=200,
                #     # gradient_clip_val=1,
                #     accelerator="gpu",
                #     devices=4,
                #     default_root_dir=save_path,
                # )
                #
                # burn_trainer.fit(
                #     model=thalamic_model,
                #     train_dataloaders=burn_train_loader,
                #     val_dataloaders=val_loader,
                # )

                save_path = set_results_path(type(thalamic_model).__name__)[0]

                trainer = Trainer(
                    max_epochs=150,
                    gradient_clip_val=10,
                    accelerator="gpu",
                    devices=1,
                    default_root_dir=save_path,
                )

                trainer.fit(
                    model=thalamic_model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )

                trainer.test(
                    thalamic_model,
                    dataloaders=DataLoader(test_set["data"], num_workers=10),
                )

                for batch_idx, batch in enumerate(val_loader):
                    thalamic_model.evaluate_training(batch, original_network=simple_model)

                pdb.set_trace()
                for batch_idx, batch in enumerate(val_loader):
                    simple_model.change_context(batch, new_context=(2, 0, 1))

                thalamic_model.save_model()
                plt.close('all')