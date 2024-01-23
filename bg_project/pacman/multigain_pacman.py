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
from itertools import product

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
    test_networks = [
        "RNNStaticBG",
        "VanillaRNN",
    ]
    network_type = "PallidalRNN"
    weight_penalties = np.logspace(-5, -1)
    weight_penalty = 0
    torch.set_float32_matmul_precision("medium")
    trial_duration = 150

    n_batches = 5

    polarity = [1]
    viscosities = np.linspace(0, 1, 10)
    masses = np.linspace(0.5, 2, 10)
    spring_constants = np.linspace(0.5, 2, 10)
    for _ in range(n_batches):

        simple_model = MultiGainPacMan(
            network=network_type,
            duration=trial_duration,
            apply_energy_penalty=("r_act",),
            output_weight_penalty=weight_penalty,
            bg_input_size=3,
        )
        dataset = PacmanDataset(
            n_samples=5000,
            masses=(1,),
            viscosity=(0,),
            polarity=(1,),
            spring_constant=(0.5,),
            trial_duration=trial_duration,
        )
        batch_size = 64

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
            max_epochs=5,
            gradient_clip_val=10,
            accelerator="gpu",
            devices=1,
            default_root_dir=save_path,
        )

        trainer.fit(
            model=simple_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        for batch_idx, batch in enumerate(val_loader):
            loss = simple_model.evaluate_training(batch)
        plt.close("all")

        for batch_idx, batch in enumerate(val_loader):
            simple_model.change_context(batch, new_context=(1, 0, 1, 2))
        pdb.set_trace()
        plt.close("all")

        for batch_idx, batch in enumerate(val_loader):
            simple_model.change_context(batch, new_context=(2, 0, 1, 0.5))
        pdb.set_trace()
        plt.close("all")

        for batch_idx, batch in enumerate(val_loader):
            simple_model.change_context(batch, new_context=(1, 2, 1, 0.5))
        pdb.set_trace()
        plt.close("all")

        simple_model.save_model()

        pdb.set_trace()

        gain_vectors = []
        contexts = []

        for mass, gain, vis, spring_k in product(
            masses, polarity, viscosities, spring_constants
        ):
            for network in test_networks:
                print(f"Mass: {mass}, polarity: {gain}, viscosity: {vis}")
                train_set, val_set, test_set = split_dataset(
                    PacmanDataset(
                        n_samples=5000,
                        masses=(mass,),
                        viscosity=(vis,),
                        polarity=(gain,),
                        spring_constant=(spring_k,),
                        trial_duration=trial_duration,
                    ),
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

                thalamic_model = MultiGainPacMan(
                    network=network_type,
                    duration=trial_duration,
                    nbg=10,
                    output_weight_penalty=0,
                    bg_input_size=3,
                    teacher_output_penalty=weight_penalty,
                    lr=1e-2,
                    polarity=gain,
                    viscosity=vis,
                    mass=mass,
                    spring_constant=spring_k,
                    original_model=simple_model.network.save_path.root,
                )

                # Transfer and freeze weights from trained network's rnn module
                transfer_network_weights(
                    thalamic_model.network,
                    simple_model.network,
                    freeze=True,
                )

                save_path = set_results_path(type(thalamic_model).__name__)[0]

                trainer = Trainer(
                    max_epochs=25,
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
                    thalamic_model.evaluate_training(
                        batch, original_network=simple_model
                    )
                    plt.pause(0.1)
                    pdb.set_trace()

                # for batch_idx, batch in enumerate(val_loader):
                #     simple_model.change_context(batch, new_context=(2, 0, -1))

                thalamic_model.save_model()
                plt.close("all")
