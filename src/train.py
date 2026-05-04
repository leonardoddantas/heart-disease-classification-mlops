import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.model import build_model


def train_model(config, train_loader, test_loader, input_dim):

    wandb.init(
        project="heart-disease-classification-mlops",
        job_type="train_model",
        name=f"lr_{config['model']['learning_rate']}_layers_{len(config['model']['hidden_sizes'])}",
        config=config,
        reinit=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    model = build_model(input_dim).to(device)

    # LOSS & OPTIMIZER
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"]
    )

    wandb.watch(model, log="all", log_freq=10)

    # EARLY STOPPING
    best_val_loss = float("inf")
    patience = config["model"]["early_stopping_patience"]
    patience_counter = 0

    # TRAIN LOOP
    for epoch in range(config["model"]["epochs"]):

        # TRAIN
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            output = model(X_batch)
            loss = criterion(output, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(X_batch)
                loss = criterion(output, y_batch)

                val_loss += loss.item() * X_batch.size(0)

                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == y_batch).sum().item()

        val_loss /= len(test_loader.dataset)
        val_acc = correct / len(test_loader.dataset)

        # LOGGING
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # EARLY STOPPING
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), "best_model.pt")

            model_artifact = wandb.Artifact(
                 f"trained_model_{wandb.run.name}",
                type="model",
                description=f"Best model at epoch {epoch}"
            )
            model_artifact.add_file("best_model.pt")
            wandb.log_artifact(model_artifact)

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load("best_model.pt"))

    return model