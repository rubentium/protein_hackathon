import argparse
import os
from glob import glob
from pathlib import Path

import torch  # type:ignore
from tqdm import tqdm

from phyloformer.model import Phyloformer
from phyloformer.data import load_alignment


def load_single_sequences(filepath):
    """
    Reads a fasta file and returns individual padded tensors for each sequence
    """
    sequences, ids = [], []

    with open(filepath, "r") as aln:
        for line in aln:
            line = line.strip()
            if line[0] == ">":
                ids.append(line[1:])
            else:
                sequences.append(line)

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    
    # Create individual tensors for each sequence
    exit_tensors = []
    for i, seq in enumerate(sequences):
        # Tokenize single sequence
        seq_tensor = tokenizer([seq], return_tensors="pt")["input_ids"]
        seq_len = seq_tensor.shape[1]
        exit_tensors.append(seq_tensor)
        

    
    return exit_tensors, ids


def vec_to_phylip(preds, ids):
    n = len(ids)
    dm = torch.zeros((n, n)).type_as(preds)
    i = torch.triu_indices(row=n, col=n, offset=1)
    dm[i[0], i[1]] = preds

    s = f"{n}\n"
    for id, row in zip(ids, dm + dm.T):
        row_s = " ".join([f"{x:.10f}" for x in row])
        s += f"{id} {row_s}\n"

    return dm + dm.T, s


def get_batch_dms(batch_preds, n):
    dms = torch.zeros((batch_preds.shape[0], n, n)).type_as(batch_preds)
    i = torch.triu_indices(row=n, col=n, offset=1)
    dms[:, i[0], i[1]] = batch_preds

    return dms + dms.transpose(-1, -2)


def has_fasta_ext(alnpath):
    """Checks if a path ends in .fa or .fasta"""
    return alnpath.lower().endswith(".fa") or alnpath.lower().endswith(".fasta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer evolutionnary distances with PhyloFormer"
    )
    parser.add_argument("weights", help="Path to model weights to use")
    parser.add_argument(
        "alndir",
        help="Path to directory containing alignments to infer",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default=None,
        required=False,
        help="Path to directory where inferred distance matrices will be written",
    )
    parser.add_argument(
        "--trees", "-t", action="store_true", help="Output NJ trees as well as matrices"
    )
    args = parser.parse_args()


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # Loading model Which means we don't need lightning anymore
    ckpt = torch.load(args.weights, map_location=device)
    params = ckpt["hyper_parameters"]
    params["device"] = device
    model = Phyloformer(**params)
    model.load_state_dict(
        {
            k.replace("model.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k != "model.seq2pair"
        },
        strict=False,
    )

    # Move model to correct place
    model = model.to(device)
    model.eval()

    # Path to dirs
    in_dir = os.path.abspath(args.alndir)
    out_dir = os.path.abspath(args.outdir)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    alnpaths = 'path/to/inference/fasta/__.fasta'  # Define this
    with torch.no_grad():
        prev_shape = None

        sequences, ids = load_single_sequences(alnpath)
        for seq, name in zip(sequences, ids):
                preds = model(seq[None, :].to(device).float())

                # feed to next model




