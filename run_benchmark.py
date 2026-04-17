import numpy as np
from collections import defaultdict
import random
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import mmh3
import urllib.request
import re
import csv
from algorithms import CountMinSketch, CountSketch, CountMinAlphaNoiseCancelled, MisraGriesAlphaQuarantine, MisraGriesExtended, SpaceSavingExtended, CountSketchAlphaOptimized
from tqdm import tqdm 

def create_real_world_nlp_dataset(stream_length=None, source_url="https://www.gutenberg.org/cache/epub/100/pg100.txt"):
    response = urllib.request.urlopen(source_url)
    text = response.read().decode('utf-8')
    data = re.findall(r'\b\w+\b', text.lower())
    if stream_length is not None:
        return data[:stream_length]
    return data

def create_skewed_dataset(stream_length, vocab_size, zipf_param=1.2):
    data = np.random.zipf(zipf_param, stream_length) % vocab_size
    return data

def create_balanced_dataset(stream_length, vocab_size):
    return np.random.randint(0, vocab_size, stream_length)

def alpha_bounded_stream_generator(base_data, alpha):
    current_freqs = defaultdict(int)
    active_elements = [] 
    
    V = 0 
    W = 0 
    
    data_idx = 0
    stream_length = len(base_data)
    
    while data_idx < stream_length or active_elements:
        can_delete = (V + 1) <= alpha * (W - 1) and len(active_elements) > 0
        
        if data_idx < stream_length and (not can_delete or random.random() > 0.85):
            item = base_data[data_idx]
            data_idx += 1
            weight = 1
            
            if current_freqs[item] == 0:
                active_elements.append(item)
            current_freqs[item] += weight
            
            V += abs(weight)
            W += weight
            yield (item, weight)
            
        elif can_delete:
            item = random.choice(active_elements)
            weight = -1
            
            current_freqs[item] += weight
            if current_freqs[item] == 0:
                active_elements.remove(item)
                
            V += abs(weight)
            W += weight
            yield (item, weight)
        else:
            break

def run_benchmark(stream_lengths, alphas, vocab_size=1000, k=20, width=50, depth=5):
    results = []
    
    # Requirement: Evaluate on diverse datasets [cite: 31, 32]
    # dataset_types = ['Balanced']
    dataset_types = ['Skewed', 'Balanced', 'Text Stream']
    
    # Pre-fetch the real-world dataset to prevent redundant HTTP requests in the loop
    max_len = max(stream_lengths)
    print(f"Fetching NLP dataset (max {max_len} tokens)...")
    full_nlp_data = create_real_world_nlp_dataset(max_len)

    for length in tqdm(stream_lengths, desc="Stream Lengths"):
        for alpha in alphas:
            for ds_type in dataset_types:
                print(f"\nEvaluating {ds_type} dataset with stream length {length} and alpha {alpha}...")
                
                # Assign base_data according to the dataset requirement category 
                if ds_type == 'Skewed':
                    base_data = create_skewed_dataset(length, vocab_size)
                elif ds_type == 'Balanced':
                    base_data = create_balanced_dataset(length, vocab_size)
                elif ds_type == 'Text Stream':
                    base_data = full_nlp_data[:length] 
                
                # Materialize stream with Strict Turnstile and a-Bounded properties
                stream = list(alpha_bounded_stream_generator(base_data, alpha))
                
                true_freqs = defaultdict(int)
                for item, weight in stream:
                    true_freqs[item] += weight
                
                # Restrict to elements where true frequency f_e >= 0 
                active_items = {key: val for key, val in true_freqs.items() if val > 0}
                if not active_items: 
                    continue 

                models = {
                    # 'Count-Min': CountMinSketch(width, depth),
                    # 'Count-Min Alpha Noise-Cancelled': CountMinAlphaNoiseCancelled(width, depth),
                    # 'Count-Sketch': CountSketch(width, depth),
                    # 'Count-Sketch Alpha Optimized': CountSketchAlphaOptimized(width, depth),
                    # 'Misra-Gries': MisraGriesExtended(k),
                    # 'MisraGriesAlphaQuarantine': MisraGriesAlphaQuarantine(k, alpha),
                    'Space-Saving': SpaceSavingExtended(k)
                }

                for name, model in models.items():
                    start_time = time.time()
                    
                    for item, weight in stream:
                        model.update(item, weight)
                        
                    exec_time = time.time() - start_time
                    
                    # Evaluate relative error [cite: 28]
                    rel_errors = []
                    for item, true_f in active_items.items():
                        est_f = model.query(item)
                        rel_errors.append(abs(est_f - true_f) / true_f)
                        
                    mean_rel_error = np.mean(rel_errors)
                    
                    # Evaluate space consumption dynamically [cite: 28]
                    space_bytes = sys.getsizeof(model)
                    if hasattr(model, 'table'):
                        space_bytes += model.table.nbytes
                    elif hasattr(model, 'counters'):
                        space_bytes += sys.getsizeof(model.counters)

                    results.append({
                        'Algorithm': name,
                        'Dataset': ds_type,
                        'Stream Length': length,
                        'Alpha': alpha,
                        'Mean Relative Error': mean_rel_error,
                        'Space (Bytes)': space_bytes,
                        'Time (s)': exec_time
                    })

    return pd.DataFrame(results)
    

if __name__ == "__main__":
    stream_lengths = [50000, 100000, 200000, 500000,]
    alphas = [1.5, 2.0, 4.0, 8.0]

    benchmark_results = run_benchmark(stream_lengths, alphas, vocab_size=100000, k=70, width=100, depth=5)
    benchmark_results.to_csv('space_saving_benchmark_results.csv', index=False)
    
    print(benchmark_results.head(50))
