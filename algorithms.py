import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import mmh3

class CountMinSketch:
    """
    Count-Min Sketch natively supports turnstile updates[cite: 22].
    """
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)

    def update(self, item, weight):
        item_str = str(item)
        for i in range(self.depth):
            idx = mmh3.hash(item_str, i) % self.width
            self.table[i, idx] += weight

    def query(self, item):
        item_str = str(item)
        min_val = float('inf')
        for i in range(self.depth):
            idx = mmh3.hash(item_str, i) % self.width
            min_val = min(min_val, self.table[i, idx])
        return max(0, min_val) # Strict turnstile guarantee: f_e >= 0 [cite: 11]

class CountSketch:
    """
    Count-Sketch natively supports turnstile updates[cite: 22].
    """
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)

    def update(self, item, weight):
        item_str = str(item)
        for i in range(self.depth):
            idx = mmh3.hash(item_str, i) % self.width
            # Sign hash determines if we add or subtract
            sign = 1 if mmh3.hash(item_str, i + self.depth) % 2 == 0 else -1
            self.table[i, idx] += weight * sign

    def query(self, item):
        item_str = str(item)
        estimates = []
        for i in range(self.depth):
            idx = mmh3.hash(item_str, i) % self.width
            sign = 1 if mmh3.hash(item_str, i + self.depth) % 2 == 0 else -1
            estimates.append(self.table[i, idx] * sign)
        return max(0, int(np.median(estimates))) # Strict turnstile guarantee [cite: 11]

class CountMinAlphaNoiseCancelled:
    """
    Enhanced Count-Min Sketch explicitly designed for the Strict Turnstile Model.
    Utilizes the global stream weight (stabilized by the alpha-bound) to perform
    active noise cancellation on hash collisions.
    """
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)
        # Track exact global weight W = sum(f_e)
        self.total_weight = 0  

    def update(self, item, weight):
        item_str = str(item)
        self.total_weight += weight
        
        for i in range(self.depth):
            idx = mmh3.hash(item_str, i) % self.width
            self.table[i, idx] += weight

    def query(self, item):
        item_str = str(item)
        adjusted_estimates = []
        
        for i in range(self.depth):
            idx = mmh3.hash(item_str, i) % self.width
            raw_val = self.table[i, idx]
            
            # The total weight distributed among the OTHER (width - 1) buckets 
            # in this row is exactly (total_weight - raw_val).
            # The expected background noise in our target bucket is the average 
            # of that remaining weight.
            expected_noise = (self.total_weight - raw_val) / max(1, (self.width - 1))
            
            # Subtract the noise to get the isolated item frequency
            adjusted_val = raw_val - expected_noise
            adjusted_estimates.append(adjusted_val)
            
        # Taking the median of noise-cancelled estimates mitigates outliers 
        # better than taking the minimum, due to the subtraction mechanics.
        final_estimate = np.median(adjusted_estimates)
        
        # Enforce the Strict Turnstile property (f_e >= 0)
        return max(0, int(round(final_estimate)))
    
class MisraGriesExtended:
    """
    Misra-Gries extended for Strict Turnstile with a-Bounded Deletion.
    Handles deletions as a primary challenge[cite: 23].
    """
    def __init__(self, k):
        self.k = k
        self.counters = defaultdict(int)

    def update(self, item, weight):
        if weight > 0:
            if item in self.counters:
                self.counters[item] += weight
            elif len(self.counters) < self.k:
                self.counters[item] = weight
            else:
                # Decrement all by weight, remove if <= 0
                keys_to_remove = []
                for k in self.counters:
                    self.counters[k] -= weight
                    if self.counters[k] <= 0:
                        keys_to_remove.append(k)
                for k in keys_to_remove:
                    del self.counters[k]
        elif weight < 0:
            # Under strict turnstile, f_e >= 0[cite: 11]. 
            # If item is tracked, apply deletion. If not, it was previously evicted.
            if item in self.counters:
                self.counters[item] += weight
                if self.counters[item] <= 0:
                    del self.counters[item]

    def query(self, item):
        return self.counters.get(item, 0)

class SpaceSavingExtended:
    """
    Space-Saving extended for Strict Turnstile.
    Handles deletions as a primary challenge[cite: 23].
    """
    def __init__(self, k):
        self.k = k
        self.counters = defaultdict(int)

    def update(self, item, weight):
        if weight > 0:
            if item in self.counters:
                self.counters[item] += weight
            elif len(self.counters) < self.k:
                self.counters[item] = weight
            else:
                # Find min counter, replace it
                min_item = min(self.counters, key=self.counters.get)
                min_val = self.counters[min_item]
                del self.counters[min_item]
                self.counters[item] = min_val + weight
        elif weight < 0:
            # Deletions in Space-Saving
            if item in self.counters:
                self.counters[item] += weight
                if self.counters[item] <= 0:
                    del self.counters[item]

    def query(self, item):
        return self.counters.get(item, 0)