import urllib.request
import re
import random
from collections import defaultdict


def fetch_gutenberg_tokens(url="https://www.gutenberg.org/cache/epub/100/pg100.txt", max_tokens=10000):
    """Fetches a text corpus and tokenizes it into words."""
    response = urllib.request.urlopen(url)
    text = response.read().decode('utf-8')
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens[:max_tokens]

def nlp_alpha_bounded_stream(base_tokens, alpha):
    """
    Simulates a stream from text tokens satisfying the Strict Turnstile Model 
    and alpha-Bounded Deletion property[cite: 4, 10, 12].
    """
    current_freqs = defaultdict(int)
    active_vocabulary = []
    
    total_absolute_volume = 0 
    total_final_weight = 0    
    
    token_idx = 0
    stream_length = len(base_tokens)
    
    while token_idx < stream_length or active_vocabulary:
        # Check if deletion maintains the alpha-property: sum(|v|) <= alpha * sum(|f|) [cite: 14]
        can_delete = (total_absolute_volume + 1) <= alpha * (total_final_weight - 1) and len(active_vocabulary) > 0
        
        # 80% chance to insert a new word from the text, 20% chance to delete an existing word
        if token_idx < stream_length and (not can_delete or random.random() > 0.20):
            # Insertion (v_t > 0) [cite: 8, 9]
            word = base_tokens[token_idx]
            token_idx += 1
            weight = 1
            
            if current_freqs[word] == 0:
                active_vocabulary.append(word)
            current_freqs[word] += weight
            
            total_absolute_volume += abs(weight)
            total_final_weight += weight
            yield (word, weight)
            
        elif can_delete:
            # Deletion (v_t < 0) [cite: 9]
            # Deleting only from active_vocabulary ensures frequency never drops below zero (Strict Turnstile) [cite: 11]
            word = random.choice(active_vocabulary)
            weight = -1
            
            current_freqs[word] += weight
            if current_freqs[word] == 0:
                active_vocabulary.remove(word)
                
            total_absolute_volume += abs(weight)
            total_final_weight += weight
            yield (word, weight)
            
        else:
            break

# Example Usage
if __name__ == "__main__":
    print("Fetching tokens...")
    gutenberg_tokens = fetch_gutenberg_tokens(max_tokens=5000)
    
    alpha_value = 2.0
    stream = nlp_alpha_bounded_stream(gutenberg_tokens, alpha_value)
    
    print(f"Generating stream with alpha={alpha_value}...\n")
    for _ in range(15):
        update = next(stream)
        action = "INSERT" if update[1] > 0 else "DELETE"
        print(f"{action:6} | Word: {update[0]:12} | Weight: {update[1]}")