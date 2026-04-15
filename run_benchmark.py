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
from algorithms import CountMinSketch, CountSketch, CountMinAlphaNoiseCancelled, MisraGriesExtended, SpaceSavingExtended
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
    dataset_types = ['Skewed', 'Balanced', 'Real-World (NLP)']
    
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
                elif ds_type == 'Real-World (NLP)':
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
                    'Count-Min': CountMinSketch(width, depth),
                    'Count-Min Alpha Noise-Cancelled': CountMinAlphaNoiseCancelled(width, depth),
                    'Count-Sketch': CountSketch(width, depth),
                    'Misra-Gries': MisraGriesExtended(k),
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


def plot_parametric_evaluation(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Error vs Alpha (using the longest stream length)
    max_length = df['Stream Length'].max()
    df_alpha = df[df['Stream Length'] == max_length]
    for alg in df['Algorithm'].unique():
        subset = df_alpha[df_alpha['Algorithm'] == alg]
        axes[0].plot(subset['Alpha'], subset['Mean Relative Error'], marker='o', label=alg)
    
    axes[0].set_title(f'Relative Error vs Alpha (N={max_length})')
    axes[0].set_xlabel('Alpha parameter')
    axes[0].set_ylabel('Mean Relative Error')
    axes[0].legend()
    
    # 2. Error vs Stream Length (using a fixed Alpha)
    fixed_alpha = df['Alpha'].median() if df['Alpha'].median() in df['Alpha'].values else df['Alpha'].iloc[0]
    df_length = df[df['Alpha'] == fixed_alpha]
    for alg in df['Algorithm'].unique():
        subset = df_length[df_length['Algorithm'] == alg]
        axes[1].plot(subset['Stream Length'], subset['Mean Relative Error'], marker='s', label=alg)
        
    axes[1].set_title(f'Relative Error vs Stream Length (Alpha={fixed_alpha})')
    axes[1].set_xlabel('Stream Length')
    axes[1].set_ylabel('Mean Relative Error')
    axes[1].legend()

    # 3. Space Consumption Comparison
    df_space = df.groupby('Algorithm')['Space (Bytes)'].mean().reset_index()
    bars = axes[2].bar(df_space['Algorithm'], df_space['Space (Bytes)'], color=['blue', 'orange', 'green', 'red'])
    axes[2].set_title('Average Space Consumption')
    axes[2].set_ylabel('Bytes')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('parametric_evaluation.png')
    

if __name__ == "__main__":
    stream_lengths = [50000, 100000, 200000]
    alphas = [1.5, 2.0, 4.0]

    benchmark_results = run_benchmark(stream_lengths, alphas, vocab_size=100000, k=10, width=20, depth=5)
    
    print(benchmark_results.head(50))
    plot_parametric_evaluation(benchmark_results)
