import pandas as pd
import numpy as np
from Bio import SeqIO
import os
from collections import Counter
from transformers import AutoTokenizer

def read_fasta(fasta_file):
    """Read MSA into SeqRecords"""
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    return seq_dict

def fata_to_matrix(seq_dict):
    """SeqRecord to numpy array"""
    records = list(seq_dict.values())
    seq_matrix = np.array([list(str(rec.seq)) for rec in records])
    return seq_matrix

def decode_sequences_to_matrix(seqs):
    """Tokens to numpy array"""
    tokenizer = AutoTokenizer.from_pretrained("/home/navasard/Phylo-rope-v2/esm2_t12_35M_UR50D")
    decoded_sequences = np.array(tokenizer.batch_decode(seqs, skip_special_tokens=True))
    return decoded_sequences

#def aafreq_percol(seq_matrix):
#    """Calculate frequency of amino acids per column"""
#    # Unique amino acids across all columns
#    aas = sorted(set(seq_matrix.flatten()))
#    # Build frequency table
#    data = []
#    for col in range(seq_matrix.shape[1]):
#        column = seq_matrix[:, col]
#        counts = Counter(column)
#        freqs = [counts.get(aa, 0) / len(column) for aa in aas]
#        data.append(freqs)
#    freq_df = pd.DataFrame(data, columns=aas).T
#    return freq_df

def aafreq_percol(seq_matrix):
    """Calculate frequency of amino acids per column using NumPy and pandas."""
    aas = np.unique(seq_matrix)
    # Create a DataFrame for easier column-wise operations
    df = pd.DataFrame(seq_matrix)
    # Calculate frequencies for each amino acid in each column
    freq_df = pd.DataFrame(
        {aa: (df == aa).sum(axis=0) / df.shape[0] for aa in aas}
    )
    freq_df = freq_df.T  # Transpose to match original output shape
    return freq_df

def mask_conserved(freq_df, rep=0.8):
    # leave columns with [rep] percent of same aa
    binary_df = (freq_df >= rep).astype(int)
    binary_cols=binary_df.sum()
    try:
        stats=binary_cols.value_counts()[1]
    except:
        stats=0
    print(f"Number of positions with {rep} of same aa: {stats}, {(stats/len(binary_cols))*100}%")
    return binary_cols

def mask_coevolved(freq_df, aa_n=2, tol=0.03):
    # Define tolerance
    rep= 1 / aa_n
    low, high = rep - tol, rep + tol
    # Initialize binary series
    binary_cols = pd.Series(0, index=freq_df.columns)

    for col in freq_df.columns:
        col_values = freq_df[col]
        # Count rows in 0.5 ± tol
        in_range = col_values.between(low, high)
        # Count rows > 0 outside the range
        others = col_values[~in_range] > 0
        if in_range.sum() == 2 and not others.any():
            binary_cols[col] = 1
        else:
            binary_cols[col] = 0
    try:
        stats=binary_cols.value_counts()[1]
    except:
        stats=0
    print(f"Number of positions with {aa_n} aa with frquency {rep} +- {tol}: {stats}, {(stats/len(binary_cols))*100}%")
    return binary_cols

def combine_masks(binary_cols_list):
    pos_encoded = pd.Series(0,index=binary_cols_list[0].index)
    for binary_cols in binary_cols_list:
        pos_encoded = pos_encoded + binary_cols
    #repeat the mask for each msa
    pos_encoded_msa=pd.concat([pos_encoded.to_frame().T] * len(seq_dict), axis=0)
    try:
        stats=(pos_encoded_msa==1).astype(int).iloc[0].value_counts()[1]
    except:
        stats=0
    print(f"Number of positions with carcass aa : {stats}, {(stats/len(pos_encoded_msa.T))*100}%")    
    return pos_encoded_msa


#fasta_file='./phylo_data/phylo_data/msa/2_50_tips.fasta'

#seq_dict = read_fasta(fasta_file)

#seq_matrix = fata_to_matrix(seq_dict)

#freq_df = aafreq_percol(seq_matrix)

#pos_conserved = mask_conserved(freq_df)

#pos_coev2 = mask_coevolved(freq_df)

#pos_coev3 = mask_coevolved(freq_df, aa_n=3)

#pos_encoded_msa = combine_masks([pos_conserved , pos_coev2 , pos_coev3])




