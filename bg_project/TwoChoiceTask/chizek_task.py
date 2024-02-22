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
    learning_type = [
        "inputs",
        "pallidal",
        "cortical",
    ]  # defines which synapse are learnable and which should be frozen

    n_epochs_base = 15  # number of training epochs for the base model
    n_epochs_finetune = 10  # number of training epochs for the fine tuned model
    nbg = 10  # number of thalamic and basal ganglia neurons

    network_type = "PallidalRNN"
    weight_penalties = np.logspace(-5, -1)
    weight_penalty = 0
    torch.set_float32_matmul_precision("medium")
    trial_duration = 150

    n_batches = 5  # number of episodes to collect

    # Define set of environment parameters to learn over
    polarity = [1]
    viscosities = np.linspace(0, 1, 10)
    masses = np.linspace(0.5, 5, 10)
    spring_constants = np.linspace(0.5, 2, 10)
    for _ in range(n_batches):

        simple_model = MultiGainPacMan(
            network=network_type,
            duration=trial_duration,
            apply_energy_penalty=("r_act",),
            output_weight_penalty=weight_penalty,
            bg_input_size=3,
            nbg=nbg,
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
            max_epochs=n_epochs_base,
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

        simple_model.save_model()

        gain_vectors = []
        contexts = []

        for mass, gain, vis, spring_k in product(
            masses, polarity, viscosities, spring_constants
        ):
            for learning_style in learning_type:
                print(
                    f"Mass: {mass}, polarity: {gain}, viscosity: {vis}, Spring Constant: {spring_k}"
                )
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
                    nbg=nbg,
                    output_weight_penalty=0,
                    bg_input_size=3,
                    teacher_output_penalty=weight_penalty,
                    lr=1e-2,
                    polarity=gain,
                    viscosity=vis,
                    mass=mass,
                    spring_constant=spring_k,
                    original_model=simple_model.network.save_path.root,
                    learning_style=learning_style,
                )

                # Transfer and freeze weights from trained network's rnn module
                transfer_network_weights(
                    thalamic_model.network,
                    simple_model.network,
                    freeze=True,
                )
                learnable_params = thalamic_model.network.learning_styles[
                    learning_style
                ]
                thalamic_model.network.swap_grad_state(grad_state=False)

                if learning_style == "inputs":
                    thalamic_model.network.swap_grad_state(
                        grad_state=True,
                        params_to_swap=[
                            thalamic_model.network.rnn.I.environment_params
                        ],
                    )
                else:
                    thalamic_model.network.swap_grad_state(
                        grad_state=True, params_to_swap=learnable_params
                    )

                save_path = set_results_path(type(thalamic_model).__name__)[0]

                trainer = Trainer(
                    max_epochs=n_epochs_finetune,
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

                # for batch_idx, batch in enumerate(val_loader):
                #     thalamic_model.evaluate_training(
                #         batch, original_network=simple_model
                #     )
                #     plt.pause(0.1)
                # pdb.set_trace()

                # for batch_idx, batch in enumerate(val_loader):
                #     simple_model.change_context(batch, new_context=(2, 0, -1))

                thalamic_model.save_model()
                plt.close("all")
