import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import mmh3
import collections

class CountSketchAlphaOptimized:
    """
    Enhanced Count-Sketch explicitly designed for the Strict Turnstile Model 
    with the alpha-Bounded Deletion property.
    """
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)
        # Track the global exact weight of the stream
        self.total_weight = 0

    def update(self, item, weight):
        item_str = str(item)
        self.total_weight += weight

        for i in range(self.depth):
            # Hash for bucket index
            idx = mmh3.hash(item_str, i) % self.width
            # Hash for sign (+1 or -1)
            sign = 1 if mmh3.hash(item_str, i + self.depth) % 2 == 0 else -1
            
            self.table[i, idx] += weight * sign

    def query(self, item):
        item_str = str(item)
        estimates = []
        
        for i in range(self.depth):
            idx = mmh3.hash(item_str, i) % self.width
            sign = 1 if mmh3.hash(item_str, i + self.depth) % 2 == 0 else -1
            # Reconstruct the estimate for this row
            estimates.append(self.table[i, idx] * sign)
            
        # Count-Sketch inherently uses the median of the independent estimates
        raw_estimate = int(np.median(estimates))
        
        # --- Explicit Alpha-Bounded & Strict Turnstile Clipping Filter ---
        # 1. Strict Turnstile constraint: The true frequency f_e can never drop below zero.
        # 2. Alpha-Bound constraint: Because total stream churn is strictly bounded, tracking the
        #    exact total_weight is reliable. Since f_e >= 0 for all elements, no single element's 
        #    frequency can mathematically exceed the total active weight of the stream.
        
        clamped_estimate = max(0, min(self.total_weight, raw_estimate))
        
        return clamped_estimate
    
class CountMinSketch:
    """
    Count-Min Sketch natively supports turnstile updates.
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
        return max(0, min_val) # Strict turnstile guarantee: f_e >= 0 

class CountSketch:
    """
    Count-Sketch natively supports turnstile updates.
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
        return max(0, int(np.median(estimates))) # Strict turnstile guarantee 

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


class MisraGriesAlphaQuarantine:
    """
    Misra-Gries extended with an alpha-scaled Quarantine Cache.
    Solves the 'double punishment' premature wipeout by temporarily holding 
    evicted elements, preventing immediate amnesia under high deletion churn.
    """
    def __init__(self, k, quarantine_ratio=0.2):
        self.k = k
        # Allocate capacity: e.g., 80% primary, 20% quarantine
        self.k_quarantine = max(1, int(k * quarantine_ratio))
        self.k_primary = max(1, k - self.k_quarantine)
        
        self.counters = collections.defaultdict(int)
        # OrderedDict acts as an O(1) FIFO queue for the quarantine buffer
        self.quarantine = collections.OrderedDict()

    def _move_to_quarantine(self, item):
        """Moves an item from primary cache to the quarantine buffer."""
        if item in self.counters:
            del self.counters[item]
            
        # Add to quarantine (or update its position to the end if already there)
        self.quarantine[item] = 0
        
        # Enforce FIFO constraint: drop the oldest "dead" item if full
        if len(self.quarantine) > self.k_quarantine:
            self.quarantine.popitem(last=False)

    def update(self, item, weight):
        if weight > 0:
            # INSERTION
            if item in self.counters:
                self.counters[item] += weight
            elif item in self.quarantine:
                # Resurrect from quarantine back to primary cache
                del self.quarantine[item]
                self.counters[item] = weight
            elif len(self.counters) < self.k_primary:
                self.counters[item] = weight
            else:
                # Capacity full: Misra-Gries decrement penalty
                keys_to_remove = []
                for key in self.counters:
                    self.counters[key] -= weight
                    if self.counters[key] <= 0:
                        keys_to_remove.append(key)
                
                # Move penalized items to quarantine instead of instantly deleting
                for key in keys_to_remove:
                    self._move_to_quarantine(key)
                    
        elif weight < 0:
            # DELETION
            if item in self.counters:
                self.counters[item] += weight
                # If deletion pushes it to zero, quarantine it
                if self.counters[item] <= 0:
                    self._move_to_quarantine(item)
            elif item in self.quarantine:
                # Item is already at 0 in quarantine; strict turnstile limits f_e >= 0.
                pass

    def query(self, item):
        return self.counters.get(item, 0)
    
class MisraGriesExtended:
    """
    Misra-Gries extended for Strict Turnstile with a-Bounded Deletion.
    Handles deletions as a primary challenge.
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
                for key in self.counters:
                    self.counters[key] -= weight
                    if self.counters[key] <= 0:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del self.counters[key]
        elif weight < 0:
            # Under strict turnstile, f_e >= 0. 
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