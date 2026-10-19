import torch
import numpy as np
import pickle
import os
import gc
from Bio.Seq import MutableSeq
from utils.utils import load_model_from_safetensors, one_hot_encode
from models.minuet import Minuet
from captum.attr import IntegratedGradients
from deeplift.dinuc_shuffle import dinuc_shuffle

def create_dinuc_baselines_worker(encoded_seq, n_baselines=100):
    """Create multiple dinucleotide-shuffled baselines"""
    baselines = []
    seq_np = encoded_seq.cpu().numpy() if hasattr(encoded_seq, 'cpu') else encoded_seq
    
    for _ in range(n_baselines):
        shuffled = dinuc_shuffle(seq_np, num_shufs=1)[0]
        baselines.append(shuffled)
    
    return np.array(baselines)

def compute_attributions_for_sequence_worker(sequence_str, model, device, n_topics=12, n_baselines=100, n_steps=50):
    """
    Compute integrated gradients attributions for a single sequence string across all topics
    """
    # Convert string sequence to one-hot encoding
    encoded_seq = one_hot_encode(MutableSeq(sequence_str))
    
    # Convert to tensor and move to device - Fix the warning
    if torch.is_tensor(encoded_seq):
        test_sample = encoded_seq.detach().clone().unsqueeze(0).to(device)
    else:
        test_sample = torch.tensor(encoded_seq, dtype=torch.float32).unsqueeze(0).to(device)
    test_sample.requires_grad_(True)
    
    # Create dinucleotide-shuffled baselines
    baselines_np = create_dinuc_baselines_worker(encoded_seq, n_baselines=n_baselines)
    baselines = torch.tensor(baselines_np, dtype=torch.float32).to(device)
    
    # Initialize integrated gradients
    ig = IntegratedGradients(model)
    
    # Store attributions for all topics
    all_topic_attributions = {}
    
    for topic_idx in range(n_topics):
        topic_attributions = []
        
        # Compute attributions across multiple baselines
        for i in range(baselines.shape[0]):
            baseline = baselines[i:i+1]
            attrs = ig.attribute(
                test_sample,
                baseline,
                target=topic_idx,
                n_steps=n_steps,
                internal_batch_size=1
            )
            topic_attributions.append(attrs.detach().cpu().numpy())
            
            # Clear GPU cache periodically
            if (i + 1) % 25 == 0:
                torch.cuda.empty_cache()
        
        # Average across baselines
        avg_attrs = np.mean(topic_attributions, axis=0).squeeze()
        all_topic_attributions[f'topic_{topic_idx}'] = avg_attrs
        
        # Clear memory after each topic
        del topic_attributions
        torch.cuda.empty_cache()
    
    # Clean up tensors
    del test_sample, baselines, ig
    torch.cuda.empty_cache()
    
    return all_topic_attributions, encoded_seq

def worker_process_sequences(gpu_id, sequences_batch, model_checkpoint_path, config_path, 
                           output_dir, batch_start_idx, n_topics=12, n_baselines=100):
    """
    Worker function that processes a batch of sequences on a specific GPU
    """
    # Initialize variables for cleanup
    model = None
    device = None
    
    try:
        # Set device for this worker
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
        
        # Load model in worker process
        model, _ = load_model_from_safetensors(model_checkpoint_path, config_path, 
                                             model_class=Minuet, set_eval=True)
        model = model.to(device)
        
        # Process each sequence in the batch
        results = []
        
        for i, (sequence_str, seq_id, metadata) in enumerate(sequences_batch):
            seq_idx = batch_start_idx + i
            
            try:
                # Compute attributions directly from sequence string
                attributions, encoded_seq = compute_attributions_for_sequence_worker(
                    sequence_str, 
                    model, 
                    device, 
                    n_topics=n_topics, 
                    n_baselines=n_baselines
                )
                
                # Prepare data for saving
                result = {
                    'seq_id': seq_id,
                    'sequence': sequence_str,
                    'encoded_seq': encoded_seq,
                    'attributions': attributions,
                    'metadata': metadata
                }
                
                results.append(result)
                
                # Clean up after each sequence
                del attributions, encoded_seq
                torch.cuda.empty_cache()
                gc.collect()
                    
            except Exception as e:
                torch.cuda.empty_cache()
                continue
        
        # Save results for this batch
        output_file = os.path.join(output_dir, f'batch_{batch_start_idx}_gpu_{gpu_id}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        return len(results)
        
    except Exception as e:
        return 0