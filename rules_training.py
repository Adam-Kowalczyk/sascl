from argparse import ArgumentParser
import os
import json

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from sascl.datasets.raven_rules_dataset import RAVENRulesDataset
from sascl.models.sascl.sascl import SASCL
from sascl.models.train_wrappers.sascl_rules_lightning import SASCLRulesLightning


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=50)


    args = parser.parse_args()

    print(f"Started {args.experiment_name}.", flush=True)

    DATA_PATH = args.data_path

    raven_train_dataset = RAVENRulesDataset(DATA_PATH, 'train', 4)
    raven_val_dataset = RAVENRulesDataset(DATA_PATH, 'val', 4)
    raven_test_dataset = RAVENRulesDataset(DATA_PATH, 'test', 4)

    print(f"Datasets created. Lengths: {len(raven_train_dataset)}, {len(raven_val_dataset)}, {len(raven_test_dataset)}", flush=True)
    
    if len(raven_train_dataset) == 0 or len(raven_val_dataset) == 0 or len(raven_test_dataset) == 0:
        print("DataLoader is of size 0!", flush=True)
        exit()

    train_loader = DataLoader(raven_train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    val_loader = DataLoader(raven_val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    test_loader = DataLoader(raven_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"Dataloaders created.", flush=True)

    model = SASCL()

    model_wr = SASCLRulesLightning(model, args.learning_rate)

    print(f"Models initialized.", flush=True)

    experiment_path = os.path.join(args.results_dir, args.experiment_name)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose = True)
    model_checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath = os.path.join(experiment_path, 'checkpoints'))
    trainer = pl.Trainer(max_epochs=args.epochs, devices=1, accelerator="auto", 
                        callbacks=[early_stop_callback, model_checkpoint_callback])

    print(f"Trainer with callbacks created.", flush=True)

    trainer.fit(model_wr, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"Training finished.", flush=True)

    results_val = trainer.validate(ckpt_path = 'best', dataloaders = val_loader)

    with open(os.path.join(experiment_path, 'train_results.json'), 'w') as f:
        json.dump(results_val , f)

    print(f"Training validation finished.", flush=True)

    results = trainer.validate(ckpt_path = 'best', dataloaders = test_loader)

    with open(os.path.join(experiment_path, 'test_results.json'), 'w') as f:
        json.dump(results, f)

    print(f"Testing finished.", flush=True)