import pdb
import torch
import matplotlib.pyplot as plt
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
    val_sampler = RandomSampler(train_set)
    train = {"data": train_set, "sampler": train_sampler}
    val = {"data": val_set, "sampler": val_sampler}

    return train, val, test_set


if __name__ == "__main__":
    duration = 300
    torch.set_float32_matmul_precision("medium")
    simple_model = GenerateSinePL(network="GMM", duration=duration)

    dataset = SineDataset(duration=duration)
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
        max_epochs=500,
        gradient_clip_val=1,
        accelerator="gpu",
        devices=1,
        default_root_dir=save_path,
    )
    trainer.fit(
        model=simple_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    trainer.test(simple_model, dataloaders=DataLoader(test_set, num_workers=10))

    # simple_model.save_model()
    # for batch_idx, batch in enumerate(val_loader):
    #     simple_model.evaluate_training(batch)
    pdb.set_trace()

    for nbg in [10]:
        thalamic_model = GenerateSinePL(
            network="GMM", nbg=nbg, n_context=2, duration=duration
        )

        # Transfer and freeze weights from trained network's rnn module
        transfer_network_weights(
            thalamic_model.network, simple_model.network, freeze=True
        )
        thalamic_model.network.swap_grad_state()

        # for param in thalamic_model.network.parameters():
        #     print(param.requires_grad, param.shape)
        # pdb.set_trace()
        # thalamic_model.network.rnn.reconfigure_u_v()

        train_set, val_set, test_set = split_dataset(
            SineDataset(amplitudes=(0.5, 1.5), frequencies=(1,), duration=duration),
            (0.6, 0.2, 0.2),
        )

        train_loader = DataLoader(
            train_set["data"],
            batch_size=batch_size,
            sampler=train_set["sampler"],
            num_workers=10,
        )
        val_loader = DataLoader(val_set["data"], batch_size=batch_size, num_workers=10)

        save_path = set_results_path(type(thalamic_model).__name__)[0]

        trainer = Trainer(
            max_epochs=400,
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

        trainer.test(thalamic_model, dataloaders=DataLoader(test_set, num_workers=10))
        thalamic_model.save_model()
        for batch_idx, batch in enumerate(val_loader):
            thalamic_model.evaluate_training(batch)
        pdb.set_trace()
