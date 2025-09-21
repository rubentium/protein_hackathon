
import abc
import torch
from tqdm import tqdm

from sedd.models.catsample import sample_categorical
from sedd.models.utils import get_score_fn

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, model, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass

class EulerPredictor(Predictor):
    def update_fn(self, model, x, t, step_size, mask=None):
        sigma, dsigma = self.noise(t)

        # torch.Size([1, 1024, 50258])
        score_fn = get_score_fn(model, train=False, sampling=True)
        score = score_fn(x, sigma)
        # score = score_fn(model, x, sigma, train=False)
        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score, mask)
        # TODO: What does sample_rate do?
        x = self.graph.sample_rate(x, rev_rate, mask)
        return x

class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, model, x, t, mask=None):
        sigma = self.noise(t)[0]

        score_fn = get_score_fn(model, train=False, sampling=True)
        score = score_fn(x, sigma)
        # score = score_fn(model, x, sigma, train=False)

        # TODO: What do these do?
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        #return probs.argmax(dim=-1)
        denoised = sample_categorical(probs)
        
        # Apply mask: if mask is provided, keep original values for masked positions
        if mask is not None:
            denoised = torch.where(mask, x, denoised)
            
        return denoised

class Sampler:
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device

    def sample(self, tokenizer, model, graph, noise, batch_size=1, steps=1024, eps=1e-5, denoise=True, projector = lambda x: x, show_intermediate=False, mask=None):
        cfg = self.cfg
        device = self.device

        predictor = EulerPredictor(graph, noise)
        denoiser = Denoiser(graph, noise)

        batch_dims = (batch_size, cfg['model']['length'])

        # This is a batch_size, seq_len tensor that starts at the limit
        # so in this case it is
        #   (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)
        # or in the case of gpt2
        #   [[50257, 50257, 50257, ..., 50257, 50257, 50257]]
        x = graph.sample_limit(*batch_dims).to(device)
        
        # If mask is provided, apply it to the initial state
        if mask is not None:
            mask = mask.to(device)
            # You can set masked positions to specific tokens (e.g., special tokens)
            # For now, we'll keep them as-is from the initial sampling

        # generate timesteps
        #   [1.0, 0.99219, 0.98438e, 0.97656 ...]
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        print(f"Sampling with {steps} steps")
        for i in tqdm(range(steps)):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(model, x, t, dt, mask)
            if show_intermediate:
                print(f"{i} @ {timesteps[i].item()}:")
                print(x)
                sentences = tokenizer.batch_decode(x)
                for sentence in sentences:
                    print(sentence[:100] + "...")

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(model, x, t, mask)

            if show_intermediate:
                sentences = tokenizer.batch_decode(x)
                for sentence in sentences:
                    print(f"Denoised:")
                    print(sentence[:100] + "...")

        sentences = tokenizer.batch_decode(x)
        return sentences