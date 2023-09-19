from functools import partial
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ignite.engine import Engine
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# Helpers
#########

ARotor_replication_fault_map = {
    "baseline": 0,
    "pitting": 1,
    "wear": 2,
    "micropitting": 3,
    "tff": 4,
}


def target_converter(targets, config, device):
    if config["data"] == "ARotor":
        return torch.tensor(targets, device=device).repeat_interleave(config["n_query"])
    elif config["data"] == "ARotor_replication":
        if config["n_way"] != 5:
            raise "`target_converter` for ARotor replication (probably) only works when all classes are in use"
        targets = [ARotor_replication_fault_map[x] for x in targets]
        return torch.tensor(targets, device=device).repeat_interleave(config["n_query"])
    else:
        raise "No `target_converter` configured for this dataset"


# TRAINER
#########


def relation_train_function_wrapper(engine, batch, config, model, optimizer, loss_fn, device, scaler):
    # Reset
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Non-blocking probably has no effect here,
    # because model(samples) is an immediate sync point
    samples = batch[0]  # * Already moved to device in collate_fn
    # Ignore actual class labels. Can be used to map episode labels to actual classes if necessary
    targets = torch.diag(torch.ones(10, device=device)).repeat_interleave(config["n_query"], dim=0)

    # Forward pass and loss calculation
    #! No AMP support
    if config["use_amp"]:
        raise "No AMP support!"

    outputs = model(samples)

    loss = loss_fn(outputs, targets)

    loss.backward()
    optimizer.step()

    # Pytorch Ignite integration
    train_loss = loss.item()
    engine.state.metrics = {
        "epoch": engine.state.epoch,
        "train_loss": train_loss,
    }

    return {"train_loss": train_loss}


def train_function_wrapper(engine, batch, config, model, optimizer, loss_fn, device, scaler):
    # Reset
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Legacy
    if config["data"] == "ARotor_old":
        samples = batch.to(device)
        targets = torch.arange(0, config["n_way"], dtype=torch.long, device=device)
        targets = targets.repeat_interleave(config["n_query"])
    # Currently important implementation
    else:
        # Non-blocking probably has no effect here,
        # because model(samples) is an immediate sync point
        # TODO Move preprocessing here?
        samples = batch[0]  # * Already moved to device in collate_fn
        # Conversion because the labels are returned as a (class, rpm, sensor) tuple
        # and we only care about the class here
        targets = list(zip(*batch[1]))[0]
        # targets = torch.tensor(list(zip(*batch[1]))[0], device=device)
        # Original targets are episode classes, not including that there are n_query queries
        # per class
        targets = target_converter(targets, config, device)

    # Forward pass and loss calculation

    # Automatic mixed precision (speed optimization)
    # https://pytorch.org/docs/stable/amp.html
    if config["use_amp"]:
        # if str(device) == "mps":
        #     raise Exception("AMP disabled for mps!")

        with torch.autocast(str(device)):
            outputs, _, _, _ = model(samples)
            loss = loss_fn(outputs, targets)
    else:
        outputs, _, _, _ = model(samples)
        loss = loss_fn(outputs, targets)

    # Backward pass
    # * Scaler should be used with AMP to not lose precision
    # * https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
    # * https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    # FIXME Necessary if using with amp. Check if working/implemented correctly
    if config["use_amp"] and scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # Pytorch Ignite integration
    train_loss = loss.item()
    engine.state.metrics = {
        "epoch": engine.state.epoch,
        "train_loss": train_loss,
    }
    return {"train_loss": train_loss}


def setup_trainer(
    config,
    model,
    optimizer,
    loss_fn,
    device,
):
    # Create scaler here to prevent it being recreated every time train_function is called
    # Scaler used with AMP
    scaler = None
    if str(device) == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    if config["model"] == "prototypical":
        train_function_ = train_function_wrapper

    elif config["model"] == "relation":
        train_function_ = relation_train_function_wrapper
    else:
        raise "WAT"

    train_function = partial(
        train_function_,
        config=config,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        scaler=scaler,
    )

    trainer = Engine(train_function)

    return trainer


# EVALUATOR
###########


@torch.no_grad()
def relation_eval_function_wrapper(engine, batch, config, model, device, split):
    model.eval()

    # Non-blocking probably has no effect here,
    # because model(samples) is an immediate sync point
    samples = batch[0]  # * Already moved to device in collate_fn
    #
    targets = torch.diag(torch.ones(10, device=device)).repeat_interleave(config["n_query"], dim=0)

    # Forward pass and loss calculation
    #! No AMP support
    if config["use_amp"]:
        raise "No AMP support!"

    outputs = model(samples)

    return outputs, targets


@torch.no_grad()
def eval_function_wrapper(engine, batch, config, model, device, split):
    model.eval()

    # Legacy
    if config["data"] == "ARotor_old":
        samples = batch.to(device)
        targets = torch.arange(0, config["n_way"], dtype=torch.long, device=device)
        targets = targets.repeat_interleave(config["n_query"])
    # Currently important implementation
    else:
        # Non-blocking probably has no effect here,
        # because model(samples) is an immediate sync point
        # TODO Move preprocessing here?
        samples = batch[0]  # * Already moved to device in collate_fn
        # Conversion because the labels are returned as a (class, rpm, sensor) tuple
        # and we only care about the class here
        # FIXME
        targets = list(zip(*batch[1]))[0]
        # targets = torch.tensor(list(zip(*batch[1]))[0], device=device)
        # Original targets are episode classes, not including that there are n_query queries
        # per class
        targets = target_converter(targets, config, device)

    # Forward pass and loss calculation

    # Automatic mixed precision (speed optimization)
    # https://pytorch.org/docs/stable/amp.html
    # ! AMP disabled for validation because can't use scaler here, which might affect stuff
    # if config["use_amp"]:
    #     if str(device) == "mps":
    #         raise Exception("AMP disabled for mps!")

    #     with torch.autocast(str(device), enabled=config["use_amp"]):
    #         outputs = model(samples)
    # else:
    outputs, support_embeddings, prototypes, query_embeddings = model(samples)

    # print(outputs.shape)
    # print(outputs)
    predictions = torch.argmax(outputs, dim=-1)
    # print(predictions)
    # print(targets.shape)
    # print(targets)

    plot_samples = False
    print_cf = False
    plot_embeddings = False
    save_embeddings = False

    if plot_samples:
        print(split)
        print(samples.shape)

        fig, axs = plt.subplots(4, 1, figsize=(20, 10))
        axs = axs.flatten()

        # print(samples[9, 0, 0, :].cpu())
        axs[0].plot(samples[9, 0, 0, :].cpu())
        axs[1].plot(samples[9, 1, 0, :].cpu())
        axs[2].plot(samples[9, -2, 0, :].cpu())
        axs[3].plot(samples[9, -1, 0, :].cpu())

        plt.tight_layout()
        plt.show()
        quit()

    if print_cf and split == "test":
        cf = confusion_matrix(targets.cpu(), predictions.cpu())
        print(cf)
        print()

    if plot_embeddings and split == "test":
        # print(prototypes.shape)
        # print(query_embeddings.shape)
        # quit()
        all_embeddings = torch.cat([prototypes.cpu().unsquueze(1), query_embeddings.cpu()], dim=0)
        tsne_embeddings = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=20).fit_transform(
            all_embeddings
        )

        tsne_prototypes = pd.DataFrame(tsne_embeddings[: config["n_way"]], columns=["x", "y"])
        tsne_queries = pd.DataFrame(tsne_embeddings[config["n_way"] :], columns=["x", "y"])

        proto_targets = np.arange(config["n_way"])
        query_targets = np.arange(config["n_way"]).repeat(config["n_query"])

        plt.figure(figsize=(20, 10))
        palette = sns.color_palette()

        sns.scatterplot(tsne_prototypes, x="x", y="y", s=40, hue=proto_targets, palette=palette, alpha=1.0, legend=True)
        sns.scatterplot(tsne_queries, x="x", y="y", s=10, hue=query_targets, palette=palette, alpha=0.8, legend=False)

        plt.tight_layout()
        plt.show()

    if save_embeddings and split == "test":
        with open("embeddings.pkl", "ab") as f:
            pickle.dump(
                {
                    "support_embeddings": support_embeddings.cpu().numpy(),
                    "prototypes": prototypes.cpu().numpy(),
                    "query_embeddings": query_embeddings.cpu().numpy(),
                },
                f,
            )

    return outputs, targets


def setup_evaluator(config, model, device, split):

    if config["model"] == "prototypical":
        eval_function_ = eval_function_wrapper

    elif config["model"] == "relation":
        eval_function_ = relation_eval_function_wrapper

    eval_function = partial(
        eval_function_,
        config=config,
        model=model,
        device=device,
        split=split,
    )

    return Engine(eval_function)
