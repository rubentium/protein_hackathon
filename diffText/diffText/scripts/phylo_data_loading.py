import lightning
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from itertools import combinations
import dendropy
import torch

WORKERS_TRAIN=4
WORKERS_VAL=2

def load_alignment(filepath):
    """
    Reads a fasta formater alignment and returns a one-hot encoded
    tensor of the MSA and the corresponding taxa label order
    """
    sequences, ids = [], []

    with open(filepath, "r") as aln:

        for line in aln:
            line = line.strip()
            if line[0] == ">":
                ids.append(line[1:])
            else:
                sequences.append(line)

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D", token="")
    seqs = tokenizer(sequences, return_tensors="pt")["input_ids"]
    return seqs, ids


def load_distance_matrix(filepath, ids):
    """
    Reads a newick formatted tree and returns a vector of the
    upper triangle of the corresponding pairwise distance matrix.
    The order of taxa in the rows and columns of the corresponding
    distance matrix is given by the `ids` input list.
    """

    distances = []

    with open(filepath, "r") as treefile:
        tree = dendropy.Tree.get(file=treefile, schema="newick")
    taxa = tree.taxon_namespace
    dm = tree.phylogenetic_distance_matrix()
    for tip1, tip2 in combinations(ids, 2):
        l1, l2 = taxa.get_taxon(tip1), taxa.get_taxon(tip2)
        distances.append(dm.distance(l1, l2))

    return torch.tensor(distances)


class PhyloDataset(Dataset):
    """
    Simple pytorch dataset that reads tree/alignment pairs
    and returns the corresponding tensor objects
    """

    def __init__(self, pairs):
        """
        pairs: List[(str,str)] = a list of (treefile, alnfile) paths
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        treefile, alnfile = self.pairs[index]
        x, ids = load_alignment(alnfile)
        y = load_distance_matrix(treefile, ids)

        return x, y

class PhyloDataModule(lightning.LightningDataModule):
    def __init__(self, train_pairs, val_pairs, batch_size):
        super().__init__()
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            dataset=PhyloDataset(self.train_pairs),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=WORKERS_TRAIN,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=PhyloDataset(self.val_pairs),
            batch_size=self.batch_size,
            num_workers=WORKERS_VAL,
            pin_memory=True,
        )