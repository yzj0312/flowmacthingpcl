"""Optimizer steps for the new framework; batches contain image/partial/missing."""

def autoencoder_step(model, missing, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model.reconstruction_loss(missing)
    loss.backward()
    optimizer.step()
    return loss.detach()


def flow_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch['partial'], batch['image'], batch['missing'])
    loss.backward()
    optimizer.step()
    return loss.detach()
