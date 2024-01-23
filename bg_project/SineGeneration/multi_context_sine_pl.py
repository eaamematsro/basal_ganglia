import pdb
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from typing import Sequence
from model_factory.networks import transfer_network_weights
from datasets.tasks import GenerateSine, GenerateSinePL, set_results_path
from datasets.loaders import SineDataset
from torch.utils.data import DataLoader, RandomSampler, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

plt.style.use("ggplot")


def split_dataset(dataset, fractions: Sequence = (0.6, 0.2, 0.2)):
    train_set, val_set, test_set = random_split(dataset, fractions)
    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(val_set)
    train = {"data": train_set, "sampler": train_sampler}
    val = {"data": val_set, "sampler": val_sampler}

    return train, val, test_set


if __name__ == "__main__":
    duration = 300
    torch.set_float32_matmul_precision("medium")
    batch_size = 64
    three_phase_training = False
    multi_phase_training = True
    network_type = "PallidalRNN"
    train_types = [
        "pallidal",
        # "full",
        "cortical",
        # "inputs",
        # "thalamocortical",
        # "corticothalamo",
    ]
    n_trials = 15
    # amplitudes = tuple(np.linspace(0.5, 2, 3).tolist())
    # frequencies = tuple(np.linspace(0.5, 2, 3).tolist())
    amplitudes = (0.5,)
    frequencies = (0.5,)
    for nneurons in [
        150,
    ]:
        for trial in range(n_trials):
            for nbg in [5, 10]:
                dataset = SineDataset(duration=duration)

                simple_model = GenerateSinePL(
                    network=network_type,
                    duration=duration,
                    nbg=nbg,
                    n_classes=25,
                    nneurons=nneurons,
                )

                train_set, val_set, test_set = split_dataset(dataset, (0.6, 0.2, 0.2))

                train_loader = DataLoader(
                    train_set["data"],
                    batch_size=batch_size,
                    sampler=train_set["sampler"],
                    num_workers=10,
                )
                val_loader = DataLoader(
                    val_set["data"], batch_size=batch_size, num_workers=10
                )

                save_path = set_results_path(type(simple_model).__name__)[0]

                trainer = Trainer(
                    max_epochs=250,
                    gradient_clip_val=1,
                    accelerator="gpu",
                    devices=1,
                    default_root_dir=save_path,
                )
                trainer.fit(
                    model=simple_model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )

                # trainer.test(
                #     simple_model, dataloaders=DataLoader(test_set, num_workers=10)
                # )
                simple_model.save_model()

                if (np.mean(simple_model.test_loss[-15:]) <= 3.5) or True:

                    # for num in range(3):
                    #     # num = 2
                    #     model_path = f"/home/elom/Documents/basal_ganglia/data/models/SineGeneration/2024-01-21/model_{num}/model.pickle"
                    #     with open(model_path, "rb") as h:
                    #         pickled_data = pickle.load(h)
                    #
                    #     simple_model = pickled_data["task"]
                    #     # try:
                    #     #     print(simple_model.network.params["train_type"], num)
                    #     # except KeyError:
                    #     #     continue
                    #
                    #     for batch_idx, batch in enumerate(val_loader):
                    #         simple_model.evaluate_training(batch)
                    # pdb.set_trace()
                    # plt.close("all")
                    # pdb.set_trace()
                    for amp, freq in product(amplitudes, frequencies):
                        if multi_phase_training:

                            # for batch_idx, batch in enumerate(val_loader):
                            #     simple_model.evaluate_training(batch)
                            # Second Training Phase #
                            for train_type in train_types:
                                if three_phase_training:
                                    amplitudes = (1,)
                                    frequencies = tuple(np.linspace(0.5, 2, 5).tolist())
                                    dataset = SineDataset(
                                        amplitudes=(1,),
                                        frequencies=(0.5, 1, 2),
                                        duration=duration,
                                    )
                                else:
                                    # amplitudes = tuple(np.linspace(0.5, 2, 5).tolist())
                                    # frequencies = tuple(np.linspace(0.5, 2, 5).tolist())
                                    amplitudes = (amp,)
                                    frequencies = (freq,)
                                    dataset = SineDataset(
                                        amplitudes=amplitudes,
                                        frequencies=amplitudes,
                                        duration=duration,
                                    )
                                thalamic_model = GenerateSinePL(
                                    network=network_type,
                                    nneurons=nneurons,
                                    nbg=nbg,
                                    n_context=2,
                                    duration=duration,
                                    n_classes=len(amplitudes) * len(frequencies),
                                    train_type=train_type,
                                    amp=amp,
                                    freq=freq,
                                )
                                thalamic_model.param_normalizers = dataset

                                # Transfer and freeze weights from trained network's rnn module
                                transfer_network_weights(
                                    thalamic_model.network,
                                    simple_model.network,
                                    freeze=True,
                                )
                                # Freeze params in model based on training conditons
                                if train_type == "full":
                                    thalamic_model.network.swap_grad_state(
                                        grad_state=True
                                    )
                                    thalamic_model.network.swap_grad_state(
                                        params_to_swap=[thalamic_model.network.Wout],
                                        grad_state=False,
                                    )
                                elif train_type == "cortical":
                                    thalamic_model.network.swap_grad_state(
                                        grad_state=False
                                    )
                                    thalamic_model.network.swap_grad_state(
                                        params_to_swap=[thalamic_model.network.rnn.J]
                                    )
                                elif train_type == "pallidal":
                                    thalamic_model.network.swap_grad_state(
                                        grad_state=False
                                    )
                                    thalamic_model.network.swap_grad_state(
                                        params_to_swap=[
                                            thalamic_model.network.rnn.Wb,
                                            thalamic_model.network.rnn.U,
                                            # thalamic_model.network.rnn.Vt,
                                        ]
                                    )
                                elif train_type == "inputs":
                                    thalamic_model.network.swap_grad_state(
                                        grad_state=False
                                    )
                                    thalamic_model.network.swap_grad_state(
                                        params_to_swap=[thalamic_model.network.rnn.I]
                                    )
                                elif train_type == "thalamocortical":
                                    thalamic_model.network.swap_grad_state(
                                        grad_state=False
                                    )
                                    thalamic_model.network.swap_grad_state(
                                        params_to_swap=[thalamic_model.network.rnn.U]
                                    )
                                elif train_type == "corticothalamo":
                                    thalamic_model.network.swap_grad_state(
                                        grad_state=False
                                    )
                                    thalamic_model.network.swap_grad_state(
                                        params_to_swap=[thalamic_model.network.rnn.Vt]
                                    )
                                train_set, val_set, test_set = split_dataset(
                                    dataset,
                                    (0.6, 0.2, 0.2),
                                )

                                thalamic_model.param_normalizers = dataset.normalizers

                                train_loader = DataLoader(
                                    train_set["data"],
                                    batch_size=batch_size,
                                    sampler=train_set["sampler"],
                                    num_workers=10,
                                )
                                val_loader = DataLoader(
                                    val_set["data"],
                                    batch_size=batch_size,
                                    num_workers=10,
                                )

                                save_path = set_results_path(
                                    type(thalamic_model).__name__
                                )[0]
                                trainer = Trainer(
                                    max_epochs=50,
                                    gradient_clip_val=1,
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
                                    dataloaders=DataLoader(test_set, num_workers=10),
                                )
                                if not three_phase_training:
                                    thalamic_model.save_model()
                                    for batch_idx, batch in enumerate(val_loader):
                                        thalamic_model.evaluate_training(batch)
                                plt.close("all")
                                if three_phase_training:
                                    # Third Training Phase #
                                    train_set, val_set, test_set = split_dataset(
                                        SineDataset(
                                            amplitudes=(0.5, 1, 1.5),
                                            frequencies=(0.5, 1, 2),
                                            duration=duration,
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
                                        val_set["data"],
                                        batch_size=batch_size,
                                        num_workers=10,
                                    )

                                    save_path = set_results_path(
                                        type(thalamic_model).__name__
                                    )[0]

                                    trainer = Trainer(
                                        max_epochs=150,
                                        gradient_clip_val=1,
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
                                        dataloaders=DataLoader(
                                            test_set, num_workers=10
                                        ),
                                    )
                                    thalamic_model.save_model()
                                    for batch_idx, batch in enumerate(val_loader):
                                        thalamic_model.evaluate_training(batch)

                        # pdb.set_trace()
