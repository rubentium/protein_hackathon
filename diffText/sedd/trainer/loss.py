import torch
import numpy as np
from sedd.models.sedd import score_fn
import code

def loss_fn(batch, model, noise, graph, train=True, t=None, perturbed_batch=None, mask=None):
    """
    Batch shape: [B, L] int. D given from graph
    mask shape: [B, L] bool. True for masked positions
    """
    sampling_eps=1e-3

    if t is None:
        t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
        
    sigma, dsigma = noise(t)
    
    if perturbed_batch is None:
        debug_vars = {**globals(), **locals()}
        #code.interact(local=debug_vars, banner="Debug console - all variables available")
        sigma_expanded = sigma[:, None].expand(batch.shape[0], batch.shape[1])  # [B, L]
        perturbed_batch = graph.sample_transition(batch, sigma[:, None], mask)

    log_score = score_fn(model, perturbed_batch, sigma_expanded, train=train, sampling=False)
    # print("log_score", log_score.shape)
    # print(log_score)
    loss = graph.score_entropy(log_score, sigma_expanded[:, None], perturbed_batch, batch, mask)
    # print("loss", loss.shape)
    # print(loss)

    loss = (dsigma[:, None] * loss).sum(dim=-1)

    return loss

def step_fn(cfg, state, batch, train=True, mask=None):
    model = state['model']
    noise = state['noise']
    graph = state['graph']
    warmup = cfg['optim']['warmup']
    accum = cfg['training']['accum']
    lr = cfg['optim']['lr']
    step = state['step']
    grad_clip = 1.

    optimizer = state['optimizer']

    if train:
        loss = loss_fn(batch, model, noise, graph, train=True, mask=mask).mean() / accum
        loss.backward()

        state['step'] += 1
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        optimizer.zero_grad()
    else:
        with torch.no_grad():
            loss = loss_fn(batch, model, noise, graph, train=False, mask=mask).mean()


    return loss
