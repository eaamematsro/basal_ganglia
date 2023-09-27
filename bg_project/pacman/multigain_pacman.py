import pdb

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
    test_networks = ["RNNStaticBG", "RNNMultiContextInput"]

    weight_penalties = np.logspace(-5, -1)
    torch.set_float32_matmul_precision("medium")
    trial_duration = 150

    for weight_penalty in weight_penalties:

        simple_model = MultiGainPacMan(
            network="VanillaRNN",
            bg_input_size=3,
            apply_energy_penalty=("r_act",),
            output_weight_penalty=weight_penalty,
            duration=trial_duration,
        )
        dataset = PacmanDataset(
            n_samples=500,
            masses=(1,),
            viscosity=(0,),
            polarity=(1,),
            trial_duration=150,
        )
        batch_size = 15

        train_set, val_set, test_set = split_dataset(dataset, (0.6, 0.2, 0.2))

        train_loader = DataLoader(
            train_set["data"],
            batch_size=batch_size,
            sampler=train_set["sampler"],
            num_workers=10,
        )
        val_loader = DataLoader(val_set["data"], batch_size=batch_size, num_workers=10)

        save_path = set_results_path(type(simple_model).__name__)[0]

        trainer = Trainer(
            max_epochs=150,
            gradient_clip_val=1,
            accelerator="gpu",
            devices=4,
            default_root_dir=save_path,
        )
        trainer.fit(
            model=simple_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        trainer.test(
            simple_model, dataloaders=DataLoader(test_set["data"], num_workers=10)
        )

        simple_model.save_model()
        for network in test_networks:
            for nbg in [10, 25, 50]:
                thalamic_model = MultiGainPacMan(
                    network=network,
                    nbg=nbg,
                    bg_input_size=3,
                    apply_energy_penalty=("r_act", "bg_act"),
                    output_weight_penalty=0,
                    energy_penalty=0,
                    duration=trial_duration,
                )

                # Transfer and freeze weights from trained network's rnn module
                transfer_network_weights(
                    thalamic_model.network, simple_model.network, freeze=True
                )

                # thalamic_model.network.rnn.reconfigure_u_v()

                # Training on an easier condition set to get better initializations

                train_set, val_set, test_set = split_dataset(
                    PacmanDataset(
                        n_samples=25, trial_duration=trial_duration, polarity=(1,)
                    ),
                    (0.6, 0.2, 0.2),
                )

                burn_train_loader = DataLoader(
                    train_set["data"],
                    batch_size=batch_size,
                    sampler=train_set["sampler"],
                    num_workers=10,
                )

                val_loader = DataLoader(
                    val_set["data"], batch_size=batch_size, num_workers=10
                )

                save_path = set_results_path(type(thalamic_model).__name__)[0]

                burn_trainer = Trainer(
                    max_epochs=200,
                    gradient_clip_val=1,
                    accelerator="gpu",
                    devices=4,
                    default_root_dir=save_path,
                )

                burn_trainer.fit(
                    model=thalamic_model,
                    train_dataloaders=burn_train_loader,
                    val_dataloaders=val_loader,
                )

                train_set, val_set, test_set = split_dataset(
                    PacmanDataset(n_samples=25, trial_duration=trial_duration),
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

                save_path = set_results_path(type(thalamic_model).__name__)[0]

                trainer = Trainer(
                    max_epochs=300,
                    gradient_clip_val=1,
                    accelerator="gpu",
                    devices=4,
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

                thalamic_model.save_model()
