from functools import partial

import torch
from ignite.engine import Engine

# Helpers
#########


def target_converter(targets, config):
    return targets.repeat_interleave(config["n_query"])


# TRAINER
#########


def train_function_wrapper(
    engine, batch, config, model, optimizer, loss_fn, device, scaler
):
    # Reset
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Non-blocking probably has no effect here,
    # because model(samples) is an immediate sync point
    samples = batch[0].to(device, non_blocking=True)
    # Conversion because the labels are returned as a (class, rpm, sensor) tuple
    # and we only care about the class here
    targets = torch.tensor(list(zip(*batch[1]))[0], dtype=torch.long, device=device)
    # Original targets are for episode classes, not including that there are n_query queries
    # per class
    targets = target_converter(targets, config)

    # Forward pass and loss calculation

    # Automatic mixed precision (speed optimization)
    # https://pytorch.org/docs/stable/amp.html
    if config["use_amp"]:
        # if str(device) == "mps":
        #     raise Exception("AMP disabled for mps!")

        with torch.autocast(str(device)):
            outputs = model(samples)
            loss = loss_fn(outputs, targets)
    else:
        outputs = model(samples)
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

    train_function_ = train_function_wrapper

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
def eval_function_wrapper(engine, batch, config, model, device):
    model.eval()

    # Non-blocking probably has no effect here,
    # because model(samples) is an immediate sync point
    samples = batch[0].to(device, non_blocking=True)
    # Conversion because the labels are returned as a (class, rpm, sensor) tuple
    # and we only care about the class here
    targets = torch.tensor(list(zip(*batch[1]))[0], dtype=torch.long, device=device)
    # Original targets are for episode classes, not including that there are n_query queries
    # per class
    targets = target_converter(targets, config)

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
    outputs = model(samples)

    return outputs, targets


def setup_evaluator(
    config,
    model,
    device,
):
    eval_function_ = eval_function_wrapper

    eval_function = partial(
        eval_function_,
        config=config,
        model=model,
        device=device,
    )

    return Engine(eval_function)
