import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import platform
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

REGISTERED_MODEL = "mnist-mlp"

def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    # Thread count changes float reduction order, so pin it. MNIST MLP is
    # small enough that single-threaded costs us almost nothing.
    torch.set_num_threads(1)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def git_is_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return out.strip() != ""


def data_md5() -> str:
    """Hash of the DVC-tracked dataset, so the run points at exact bytes."""
    with open("data.dvc") as fh:
        meta = yaml.safe_load(fh)
    return meta["outs"][0]["md5"]

class MLP(nn.Module):
    def __init__(self, hidden1: int, hidden2: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 10),
        )

    def forward(self, x):
        return self.net(x)

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return loss_sum / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden1", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=128)
    p.add_argument("--note", type=str, default="", help="free-text note for the run")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()

    if git_is_dirty() and not args.allow_dirty:
        sys.exit(
            "Working tree is dirty. Commit your changes first so the run's "
            "git_commit tag actually identifies the code that produced it.\n"
            "(Override with --allow-dirty if you know what you're doing.)"
        )

    # CPU only. Forcing this removes GPU model/driver as a source of drift
    # between Partner A's machine and Partner B's.
    device = torch.device("cpu")
    g = set_seed(args.seed)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # download=False on purpose: the data must come from `dvc checkout`.
    train_ds = datasets.MNIST("data", train=True, download=False, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=False, transform=tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0)

    model = MLP(args.hidden1, args.hidden2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_experiment("mnist-mlp-repro")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden1": args.hidden1,
                "hidden2": args.hidden2,
                "optimizer": "Adam",
                "device": str(device),
                "torch_threads": torch.get_num_threads(),
            }
        )
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "data_md5": data_md5(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "platform": platform.platform(),
                "deterministic": "True",
                "note": args.note,
            }
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            running, seen = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item() * y.size(0)
                seen += y.size(0)

            train_loss = running / seen
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc},
                step=epoch,
            )
            print(
                f"epoch {epoch}  train_loss={train_loss:.6f}  "
                f"val_loss={val_loss:.6f}  val_acc={val_acc:.6f}"
            )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})
        print(f"\nFINAL  test_loss={test_loss:.6f}  test_accuracy={test_acc:.6f}")

        # Reference metrics on disk, so Partner B can diff without opening the UI.
        results = {
            "run_id": run.info.run_id,
            "git_commit": git_commit(),
            "data_md5": data_md5(),
            "seed": args.seed,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        with open("metrics.json", "w") as fh:
            json.dump(results, fh, indent=2)

        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("environment.yml")
        mlflow.log_artifact("train.py")
        mlflow.log_artifact("data.dvc")

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL,
        )
        print(f"\nrun_id: {run.info.run_id}")

    # Promote the version we just registered to Staging.
    client = MlflowClient()
    latest = max(
        client.search_model_versions(f"name='{REGISTERED_MODEL}'"),
        key=lambda v: int(v.version),
    )
    client.transition_model_version_stage(
        name=REGISTERED_MODEL, version=latest.version, stage="Staging"
    )
    print(f"registered {REGISTERED_MODEL} v{latest.version} -> Staging")


if __name__ == "__main__":
    main()
