import os
import torch
import numpy as np

from Bio.Seq import Seq, MutableSeq
from itertools import product

def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [1/4, 1/4, 1/4, 1/4],
               'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [1/4, 1/4, 1/4, 1/4]}
    
    # Handle both single sequences and lists of sequences
    if isinstance(sequence, (list, tuple)):
        # Multiple sequences
        one_hot_encoded = []
        for seq in sequence:
            seq_encoded = [mapping[base] for base in seq]
            one_hot_encoded.append(seq_encoded)
        return torch.tensor(one_hot_encoded, dtype=torch.float32)
    else:
        # Single sequence
        one_hot_encoded = [mapping[base] for base in sequence]
        return torch.tensor(one_hot_encoded, dtype=torch.float32)

# def one_hot_encode(sequence):
#     mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [1/4, 1/4, 1/4, 1/4],
#                'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [1/4, 1/4, 1/4, 1/4]}
#     one_hot_encoded = [mapping[base] for seq in sequence for base in seq]
#     return torch.tensor(one_hot_encoded, dtype=torch.float32)
    
def decode_one_hot(one_hot_sequences):
    """
    Decode one-hot encoded sequences back to DNA strings.
    
    Args:
        one_hot_sequences (torch.Tensor or list): Tensor of shape (batch_size, seq_length, 4)
                                                 or (seq_length, 4) for single sequence,
                                                 or list of one-hot encoded sequences
    
    Returns:
        list or str: List of DNA sequences if batch input, single string if single sequence
    """
    mapping = ['A', 'C', 'G', 'T']
    
    # Handle list input
    if isinstance(one_hot_sequences, list):
        # Convert list to tensor first
        one_hot_sequences = torch.tensor(one_hot_sequences, dtype=torch.float32)
    
    if one_hot_sequences.dim() == 2:
        # Single sequence: shape (seq_length, 4)
        decoded_sequence = ''.join(mapping[torch.argmax(base).item()] for base in one_hot_sequences)
        return decoded_sequence
    elif one_hot_sequences.dim() == 3:
        # Batch of sequences: shape (batch_size, seq_length, 4)
        decoded_sequences = []
        for seq in one_hot_sequences:
            decoded_seq = ''.join(mapping[torch.argmax(base).item()] for base in seq)
            decoded_sequences.append(decoded_seq)
        return decoded_sequences
    else:
        raise ValueError("Input must be 2D (single sequence) or 3D (batch of sequences).")

def reverse_complement(x):
    # Reverse the sequence batch-wise
    x = torch.flip(x, dims=[1])  # x is expected to be of shape (batch_size, seq_length, 4)
    x = torch.stack([x[:, :, 3], x[:, :, 2], x[:, :, 1], x[:, :, 0]], dim=-1)  # Complement the sequence (A<->T, C<->G)
    return x

def generate_seq(seq_length=500, gc_content=0.4, immutable=False):
    """
    Generates a random DNA sequence with a specified GC content.
    """
    gc_count = int(seq_length * gc_content)
    at_count = seq_length - gc_count
    sequence = np.random.choice(['G', 'C'], gc_count).tolist() + np.random.choice(['A', 'T'], at_count).tolist()
    np.random.shuffle(sequence)
    seq_str = ''.join(sequence)
    return Seq(seq_str) if immutable else MutableSeq(seq_str)

def saturated_mutagenesis(sequence, positions=None):
    """
    Given a single sequence, generate all possible single point mutations and outputs a list.
    Input sequence should be a MutableSeq object.
    
    Args:
        sequence (MutableSeq): The input sequence to mutate.
        positions (list, optional): A list of positions (0-based index) to mutate. 
                                    If None, all positions will be mutated.
    
    Returns:
        list: A list of mutated sequences.
    """
    nucleotides = ['A', 'T', 'C', 'G']
    seqs = []
    
    # If positions are not specified, mutate all positions
    if positions is None:
        positions = range(len(sequence))
    
    for i in positions:
        original_nucleotide = sequence[i]
        for nucleotide in nucleotides:
            if nucleotide != original_nucleotide:
                # Create a copy of the sequence to avoid modifying the original
                new_seq = sequence[:]
                new_seq[i] = nucleotide
                seqs.append(new_seq)
    
    return seqs

def get_synonymous_codons():
    """Return dictionary of synonymous codons for each amino acid"""
    codon_table = {
        'A': ['GCT', 'GCC', 'GCA', 'GCG'],
        'C': ['TGT', 'TGC'],
        'D': ['GAT', 'GAC'],
        'E': ['GAA', 'GAG'],
        'F': ['TTT', 'TTC'],
        'G': ['GGT', 'GGC', 'GGA', 'GGG'],
        'H': ['CAT', 'CAC'],
        'I': ['ATT', 'ATC', 'ATA'],
        'K': ['AAA', 'AAG'],
        'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
        'M': ['ATG'],
        'N': ['AAT', 'AAC'],
        'P': ['CCT', 'CCC', 'CCA', 'CCG'],
        'Q': ['CAA', 'CAG'],
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'],
        'V': ['GTT', 'GTC', 'GTA', 'GTG'],
        'W': ['TGG'],
        'Y': ['TAT', 'TAC'],
        '*': ['TAA', 'TAG', 'TGA']
    }
    return codon_table

def saturated_synonymous_substitution(sequence, start_pos=0, end_pos=None, 
                                     max_variants=10000, preserve_original=True):
    """
    Generate saturated synonymous substitutions for a coding sequence.
    
    Parameters:
    -----------
    sequence : str
        DNA sequence (must be divisible by 3)
    start_pos : int
        Start position for mutagenesis (default: 0)
    end_pos : int or None
        End position for mutagenesis (default: None, meaning end of sequence)
    max_variants : int
        Maximum number of variants to generate (default: 10000)
    preserve_original : bool
        Include original sequence in output (default: True)
    
    Returns:
    --------
    dict : Dictionary with variant names as keys and sequences as values
    """
    
    sequence = sequence.upper()
    
    if len(sequence) % 3 != 0:
        raise ValueError(f"Sequence length must be divisible by 3. Length: {len(sequence)}")
    
    if end_pos is None:
        end_pos = len(sequence)
    
    # Get codon table
    codon_table = get_synonymous_codons()
    
    # Extract region to mutate
    prefix = sequence[:start_pos]
    mutagenesis_region = sequence[start_pos:end_pos]
    suffix = sequence[end_pos:]
    
    # Split into codons
    codons = [mutagenesis_region[i:i+3] for i in range(0, len(mutagenesis_region), 3)]
    
    # Translate to amino acids
    seq_obj = Seq(mutagenesis_region)
    amino_acids = str(seq_obj.translate())
    
    # Get synonymous options for each codon
    synonymous_options = []
    for aa in amino_acids:
        if aa in codon_table:
            synonymous_options.append(codon_table[aa])
        else:
            raise ValueError(f"Unknown amino acid: {aa}")
    
    # Calculate total number of possible variants
    total_combinations = 1
    for options in synonymous_options:
        total_combinations *= len(options)
    
    print(f"Total possible synonymous variants: {total_combinations:,}")
    
    if total_combinations > max_variants:
        print(f"Warning: Limiting to {max_variants:,} random variants")
        # Generate random sample
        import random
        variants = {}
        attempts = 0
        max_attempts = max_variants * 10
        
        while len(variants) < max_variants and attempts < max_attempts:
            # Generate random variant
            new_codons = [random.choice(options) for options in synonymous_options]
            new_seq = prefix + ''.join(new_codons) + suffix
            
            if new_seq not in variants.values():
                variant_name = f"variant_{len(variants)+1:05d}"
                variants[variant_name] = new_seq
            
            attempts += 1
    else:
        # Generate all combinations
        variants = {}
        for idx, codon_combination in enumerate(product(*synonymous_options)):
            new_seq = prefix + ''.join(codon_combination) + suffix
            variant_name = f"variant_{idx+1:05d}"
            variants[variant_name] = new_seq
    
    # Add original sequence if requested
    if preserve_original:
        variants = {"original": sequence, **variants}
    
    print(f"Generated {len(variants)} variants (including original: {preserve_original})")
    
    return variants

def saturated_synonymous_per_codon(sequence, start_pos=0, end_pos=None):
    """
    Generate one variant per codon with all synonymous substitutions for that codon.
    More manageable for long sequences.
    
    Returns:
    --------
    dict : Dictionary with variant names and sequences
    """
    
    sequence = sequence.upper()
    
    if len(sequence) % 3 != 0:
        raise ValueError(f"Sequence length must be divisible by 3. Length: {len(sequence)}")
    
    if end_pos is None:
        end_pos = len(sequence)
    
    codon_table = get_synonymous_codons()
    
    prefix = sequence[:start_pos]
    mutagenesis_region = sequence[start_pos:end_pos]
    suffix = sequence[end_pos:]
    
    codons = [mutagenesis_region[i:i+3] for i in range(0, len(mutagenesis_region), 3)]
    seq_obj = Seq(mutagenesis_region)
    amino_acids = str(seq_obj.translate())
    
    variants = {"original": sequence}
    
    # For each codon position
    for codon_idx, (original_codon, aa) in enumerate(zip(codons, amino_acids)):
        if aa not in codon_table:
            continue
        
        synonymous_codons = codon_table[aa]
        
        # Skip if only one option (no synonyms)
        if len(synonymous_codons) <= 1:
            continue
        
        # Create variant for each synonymous codon
        for syn_codon in synonymous_codons:
            if syn_codon == original_codon:
                continue  # Skip original
            
            # Build new sequence
            new_codons = codons.copy()
            new_codons[codon_idx] = syn_codon
            new_seq = prefix + ''.join(new_codons) + suffix
            
            variant_name = f"codon_{codon_idx+1:04d}_{original_codon}_to_{syn_codon}"
            variants[variant_name] = new_seq
    
    print(f"Generated {len(variants)-1} single-codon variants")
    
    return variants

# def directed_evolution(initial_seq, model, device, objective_func, num_rounds=8, verbose=True):
#     """
#     Perform directed evolution using saturated mutagenesis and model evaluation.
    
#     Args:
#         initial_seq (MutableSeq): Initial sequence to evolve
#         model: Trained model for evaluation
#         device: PyTorch device (cuda/cpu)
#         objective_func (callable): Function that takes delta array and returns objective score
#         num_rounds (int): Number of evolution rounds to perform
#         verbose (bool): Whether to print progress information
    
#     Returns:
#         dict: Contains evolution history including sequences, evaluations, and best indices
#     """
    
#     # Initialize tracking variables
#     evolution_history = {
#         'sequences': [initial_seq[:]], # Store copy of initial sequence
#         'evaluations': [],
#         'best_indices': [],
#         'objective_scores': []
#     }
    
#     # Get initial evaluation - fix the encoding issue
#     encoded_seq = one_hot_encode(str(initial_seq)).unsqueeze(0)  # Add batch dimension
#     encoded_seq = encoded_seq.to(device)
    
#     with torch.no_grad():
#         logits = model(encoded_seq)
#         eval_initial = torch.sigmoid(logits)
    
#     eval_initial = eval_initial.cpu().numpy()
#     evolution_history['evaluations'].append(eval_initial[0])
    
#     if verbose:
#         print(f"Initial evaluation: {eval_initial[0]}")
#         print(f"Starting directed evolution for {num_rounds} rounds...")
    
#     current_seq = initial_seq[:]
#     current_eval = eval_initial
    
#     # Evolution rounds
#     for round_num in range(1, num_rounds + 1):
#         if verbose:
#             print(f"\nRound {round_num}:")
        
#         # Generate all single-point mutants
#         mutant_seqs = saturated_mutagenesis(current_seq)
        
#         # Evaluate all mutants - fix the encoding issue here too
#         encoded_mutants = []
#         for seq in mutant_seqs:
#             encoded_mutants.append(one_hot_encode(str(seq)))
        
#         sequences_tensor = torch.stack(encoded_mutants).to(device)
        
#         with torch.no_grad():
#             logits = model(sequences_tensor)
#             evaluations = torch.sigmoid(logits)
        
#         evaluations = evaluations.cpu().numpy()
        
#         # Calculate objective scores
#         delta = evaluations - current_eval
#         objective_scores = np.array([objective_func(delta[i]) for i in range(len(delta))])
        
#         # Find best mutant
#         best_index = np.argmax(objective_scores)
#         best_seq = mutant_seqs[best_index]
#         best_eval = evaluations[best_index]
#         best_objective_score = objective_scores[best_index]
        
#         # Store results
#         evolution_history['sequences'].append(best_seq[:])
#         evolution_history['evaluations'].append(best_eval)
#         evolution_history['best_indices'].append(best_index)
#         evolution_history['objective_scores'].append(best_objective_score)
        
#         if verbose:
#             print(f"  Best mutant index: {best_index}")
#             print(f"  Objective score improvement: {best_objective_score:.6f}")
#             print(f"  Best evaluation: {best_eval}")
        
#         # Update for next round
#         current_seq = best_seq[:]
#         current_eval = best_eval.reshape(1, -1)
    
#     return evolution_history

# def create_objective_function(target_topic=None, penalize_topics=None):
#     """
#     Create an objective function for directed evolution.
    
#     Args:
#         target_topic (int, optional): Index of topic to maximize (0-indexed)
#                                      If None, no topic is specifically promoted
#         penalize_topics (list, optional): List of topic indices to penalize (0-indexed)
#                                          If None and target_topic is specified, penalizes all topics except target
#                                          If None and target_topic is None, no penalties applied
    
#     Returns:
#         callable: Objective function that takes delta array and returns score
#     """
#     if target_topic is not None and penalize_topics is None:
#         # Default behavior: penalize all topics except the target
#         penalize_topics = [i for i in range(11) if i != target_topic]
#     elif penalize_topics is None:
#         # If no target and no penalties specified, set empty penalty list
#         penalize_topics = []
    
#     def objective_func(delta_row):
#         target_score = delta_row[target_topic] if target_topic is not None else 0
#         penalty_score = sum(delta_row[i] for i in penalize_topics) if penalize_topics else 0
#         return target_score - penalty_score
    
#     return objective_func
    
# def plot_evolution_heatmap(evolution_history, title="Topic Probabilities Across Evolution Rounds"):
#     """
#     Plot heatmap of topic probabilities across evolution rounds.
    
#     Args:
#         evolution_history (dict): Output from directed_evolution function
#         title (str): Title for the plot
#     """
#     import matplotlib.pyplot as plt
    
#     evolution_data = np.array(evolution_history['evaluations'])
    
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(evolution_data.T, 
#                 annot=True, 
#                 fmt='.3f',
#                 cmap='RdYlBu_r',
#                 xticklabels=[f'Round {i}' for i in range(len(evolution_data))],
#                 yticklabels=[f'Topic {i+1}' for i in range(evolution_data.shape[1])],
#                 cbar_kws={'label': 'Topic Probability'})
    
#     plt.title(title)
#     plt.xlabel('Evolution Round')
#     plt.ylabel('Topic Categories')
#     plt.tight_layout()
#     plt.show()

def load_model_from_safetensors(checkpoint_path, config_path, model_class, set_eval=True):
    """
    Load model from Accelerate-saved SafeTensors format
    Adapted from evaluation notebook in DeepMSN repository.
    """
    
    import yaml
    from safetensors.torch import load_file
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = model_class(config=config)
    
    # Load from SafeTensors
    safetensors_path = os.path.join(checkpoint_path, 'model.safetensors')
    
    if os.path.exists(safetensors_path):
        # Load the state dict from SafeTensors
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {safetensors_path}")
    else:
        raise FileNotFoundError(f"SafeTensors file not found at {safetensors_path}")
    
    # Set to eval mode and move to device
    if set_eval:
        model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, config


def load_seqs_dict_from_fasta(fasta_path):
    """Load sequences from a FASTA file into a dictionary"""
    from Bio import SeqIO
    seqs_dict = {}
    
    with open(fasta_path, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seqs_dict[record.id] = str(record.seq)
    
    return seqs_dict