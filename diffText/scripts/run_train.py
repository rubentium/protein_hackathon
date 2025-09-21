
import os
import os.path
from os import listdir
import code
import yaml
import sys

import torch
import yaml
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM

import argparse

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")

# Check if sedd directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
sedd_path = os.path.join(script_dir, '..', 'sedd')
print(f"\nLooking for sedd at: {os.path.abspath(sedd_path)}")
print(f"SEDD directory exists: {os.path.exists(sedd_path)}")

if os.path.exists(sedd_path):
    print("Contents of sedd directory:")
    for item in os.listdir(sedd_path):
        print(f"  {item}")

# Add the parent directory to Python path so we can import sedd
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"\nAdded to Python path: {parent_dir}")

from sedd.models.noise import LogLinearNoise
from sedd.models.sedd import SEDD
from sedd.models.sampler import Sampler
from sedd.models.graph import AbsorbingGraph
from sedd.trainer.trainer import Trainer
from sedd.eval.evaluator import Evaluator
from transformers import GPT2TokenizerFast

from aim import Run

from phylo_data_loading import PhyloDataset

# from sedd.models.simple_sedd import SEDD
from torch.utils.data import DataLoader

def print_devices(device):
    if torch.cuda.is_available():
        print("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        print("WARNING: Using device {}".format(device))
    print(f"Using device: {device}")
    print(f"Found {os.cpu_count()} total number of CPUs.")


def choose_data(train_alignments, val_alignments):
    """Find and select training and validation tree/MSA pairs"""
    # Choose training and validation examples

    train_path = os.listdir(train_alignments)
    val_path = os.listdir(val_alignments)

    return train_path, val_path


def main():
    args = argparse.ArgumentParser(description="Train SEDD")
    args.add_argument("--cfg", type=str, default="configs/config.yaml")
    args.add_argument("--output", type=str, default="output")
    args.add_argument("--repo", type=str, default="ox/SEDD_dev")
    args.add_argument("--msa", default=False, help="Add the multiple sequence alignment data", action="store_true")
    args.add_argument("--foundation", default=False, help="Add the amino acid foundation predictions", action="store_true")
    args.add_argument(
        "--train-alignments",
        "-a",
        default="/home/noxatras/Documents/hackathon/phylo_data/msa",
        help="Directory with training alignments",
    )
    args.add_argument(
        "--val-alignments",
        "-A",
        default="/home/noxatras/Documents/hackathon/phylo_data/msa",
        help="Directory with validation alignments",
    )
    args = args.parse_args()

    if args.cfg and os.path.exists(args.cfg):
        with open(args.cfg, 'r') as f:
            yaml_args = yaml.safe_load(f)
        
        # Update args with YAML values (YAML values override defaults but not command line)
        for key, value in yaml_args.items():
            if hasattr(args, key):
                # Only override if the argument wasn't explicitly set on command line
                if getattr(args, key) == args.__dict__.get(key):  # Check if it's the default
                    setattr(args, key, value)
            else:
                # Add new arguments from YAML
                setattr(args, key, value)
    
    print(f"Using configuration: {args}")

    train_paths, val_paths = choose_data(args.train_alignments, args.val_alignments)




    # load in tokenizer
    # tokenizer = OxTokenizer()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = '' # make sure we pad with absorbing token

    with open(args.cfg, 'r') as f:
        cfg = yaml.full_load(f)

    cfg['tokens'] = tokenizer.vocab_size
    cfg['data'] = {}
    cfg['data']['remote_repo'] = args.repo
    cfg['training']['output_dir'] = args.output

    print(cfg)

    work_dir = cfg['training']['output_dir']

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    print(work_dir)
    print(cfg)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print_devices(device)



    # build token graph
    graph = AbsorbingGraph(tokenizer.vocab_size)

    # build score model
    score_model = SEDD(cfg, tokenizer.vocab_size).to(device)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    print(f"Number of parameters in the model: {num_parameters}")

    # train_ds = DataLoader(OpenSubtitlesDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=10_000), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    # eval_ds = DataLoader(OpenSubtitlesDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=128))

    
    # train_ds = DataLoader(BabyNamesDataset(tokenizer, seq_len=cfg['model']['length']), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    train_ds = DataLoader(PhyloDataset(train_paths, args.train_alignments), batch_size=cfg['training']['batch_size'], shuffle=True)
    eval_ds = DataLoader(PhyloDataset(val_paths, args.val_alignments), batch_size=cfg['training']['batch_size'], shuffle=True)

    # train_ds = DataLoader(ABCDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=10000), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    # eval_ds = DataLoader(ABCDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=128))

    noise = LogLinearNoise().to(device)

    run = Run()
    run["hparams"] = cfg

    def eval(state):
        evaluator = Evaluator(eval_ds, run, cfg, device=device)
        return evaluator.evaluate(state)

    def sample(state):
        step = state['step']
        model = state['model']
        graph = state['graph']
        noise = state['noise']
        code.interact(local=locals())

        sampler = Sampler(cfg)
        texts = sampler.sample(tokenizer, model, graph, noise, steps=128, batch_size=cfg['eval']['batch_size'], mask=args.foundation)

        file_name = os.path.join(sample_dir, f"sample.txt")
        with open(file_name, 'w') as file:
            for sentence in texts:
                file.write(sentence + "\n")
                file.write("="*80 + "\n")


    trainer = Trainer(
        run,
        score_model,
        graph,
        noise,
        cfg,
        eval_callback=eval,
        sample_callback=sample,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    trainer.train(train_ds)


if __name__ == "__main__":
    main()