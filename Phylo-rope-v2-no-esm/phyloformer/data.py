from itertools import combinations

import dendropy
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

from msa_to_carcass import *


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

    tokenizer = AutoTokenizer.from_pretrained("/home/navasard/Phylo-rope-v2/esm2_t12_35M_UR50D")
    seqs = tokenizer(sequences, return_tensors="pt")["input_ids"]
    return seqs, ids, sequences


#def load_distance_matrix(filepath, ids):
#    """
#    Reads a newick formatted tree and returns a vector of the
#    upper triangle of the corresponding pairwise distance matrix.
#    The order of taxa in the rows and columns of the corresponding
#    distance matrix is given by the `ids` input list.
#    """
#
#    distances = []
#
#    with open(filepath, "r") as treefile:
#        tree = dendropy.Tree.get(file=treefile, schema="newick")
#    taxa = tree.taxon_namespace
#    dm = tree.phylogenetic_distance_matrix()
#    for tip1, tip2 in combinations(ids, 2):
#        l1, l2 = taxa.get_taxon(tip1), taxa.get_taxon(tip2)
#        distances.append(dm.distance(l1, l2))
#
#    return torch.tensor(distances)

def load_carcass(msa):
    """
    Encode MSA based on set of rules.
    Output: Columns = amino acid positions , Rows = MSA, Values 1-> pass one or more rules
    """

    seq_matrix = np.array(msa)

    freq_df = aafreq_percol(seq_matrix)

    pos_conserved = mask_conserved(freq_df)

    pos_coev2 = mask_coevolved(freq_df)

    pos_coev3 = mask_coevolved(freq_df, aa_n=3)

    pos_encoded_msa = combine_masks([pos_conserved , pos_coev2 , pos_coev3])

    pos_encoded_msa_list = pos_encoded_msa.values.tolist()

    return torch.tensor(pos_encoded_msa_list)



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
        x, ids, msa = load_alignment(alnfile)
        ###y = load_distance_matrix(treefile, ids)
        y = load_carcass(msa)

        return x, y
