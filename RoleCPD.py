import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from collections import Counter
import networkx as nx

class RoleCPD:
    def __init__(self, n_players=10, switch_threshold=0.7):
        """
        Initialize Role Change-Point Detector

        Args:
            n_players (int, optional): Number of outfield players. Defaults to 10.
            switch_threshold (float, optional): Threshold to filter abnormal events such as set-pieces. Defaults to 0.7.
        """
        self.N = n_players
        self.switch_threshold = switch_threshold
    
    def hamming_distance(self, permutation1, permutation2):
        """
        Compute normalized Hamming distance between two permutations.

        Args:
            permutation1 (list): First permutation.
            permutation2 (list): Second permutation.

        Returns:
            float: Normalized Hamming distance.
        """
        return np.sum(permutation1 != permutation2) / self.N
    
    def compute_switch_rate_sequence(self, permutations):
        """
        Compute the switch rate relative to the global mode to detect set-pieces.
        Args:
            permutations (list): List of permutations over time.
        
        Returns:
            np.array: Array of switch rates.
        """
        # Most frequent permutation
        tuple_permutations = [tuple(p) for p in permutations]
        most_common = np.array(Counter(tuple_permutations).most_common(1)[0][0])

        rates = []
        for perm in permutations:
            rate = self.hamming_distance(perm, most_common)
            rates.append(rate)
        
        return np.array(rates)
    
    def graph_based_scan_statistic(self, valid_permutations):
        """
        Replicates the Scan Statistic R(t) from graph-based CPD
        1. Build similarity graph (k-NN) based on Hamming distances
        2. Count edges crossing the potential cut point t.

        Args:
            valid_permutations (list or np.array): List or array of shape (T, N) containing valid role permutations.

        Returns:
            list: List of detected change points.
        """
        T = len(valid_permutations)
        if T < 10: return []

        # Compute pairwise Hamming distances
        pairwise_distances = squareform(pdist(valid_permutations, metric=self.hamming_distance))

        # Build k-NN graph: we will use k=5
        G = nx.Graph()
        for i in range(T):
            nearest = np.argsort(pairwise_distances[i])[1:6]  # skip self
            for neighbor in nearest:
                G.add_edge(i, neighbor)

        # Compute scan statistic R(t)
        scan_statistics = []

        # We only scan the middle 80% to avoid edge effects
        buffer = max(10, int(T * 0.1))
        search_range = range(buffer, T - buffer)

        for t in search_range:
            # Count edges crossing the cut at t
            cut_edges = 0
            nodes_left = set(range(0, t))
            nodes_right = set(range(t, T))

            for u, v in G.edges():
                if (u in nodes_left and v in nodes_right) or (u in nodes_right and v in nodes_left):
                    cut_edges += 1
            
            scan_statistics.append(cut_edges)

        # Find best t
        best_t = search_range[np.argmin(scan_statistics)]
        return [best_t]
    
    def decompose_cycles(self, permutation, canonical_roles):
        """
        Decomposes a permutation into disjoint cycles.

        Args:
            permutation (list or np.array): A permutation representing roles.
            canonical_roles (list or np.array): The canonical roles to compare against.
        
        Returns:
            list: List of cycles, each cycle is a list of indices.
        """
        checked = set()
        cycles = []

        for i in range(len(permutation)):
            if i in checked:
                continue

            current = i
            path = [current]
            while True:
                next_role = permutation[current]

                if next_role == i:
                    break

                if next_role in checked:
                    break

                path.append(next_role)
                current = next_role
            
            for node in path:
                checked.add(node)

            if len(path) > 1:
                cycles.append(tuple(path))
        
        return cycles
    
    def analyze_messi_effect(self, permutations, messi_role_index):
        """
        Experiment to prove Messi's low stationarity.
        1. Calculate percentage of frames where Messi is not in his instructed role.
        2. Find the most frequent cycle involving Messi.

        Args:
            permutations (): _description_
            messi_role_index (): _description_
        
        Returns:
        """
        drift_count = 0
        cycle_counts = Counter()

        base_identity = np.arange(self.N)

        for perm in permutations:
            # Check if Messi drifted
            if perm[messi_role_index] != base_identity[messi_role_index]:
                drift_count += 1

                # Find the specific cycle involving Messi
                cycles = self.decompose_cycles(perm, base_identity)
                for cycle in cycles:
                    if messi_role_index in cycle:
                        cycle_counts[cycle] += 1
            
        return drift_count / len(permutations), cycle_counts.most_common(5)

    def fit_predict(self, permutations):
        """
        Fit the model and predict change points in a sequence of role permutations.

        Args:
            permutations (list or np.array): List or array of shape (T, N) containing role indices.
        
        Returns:
            change_points (list): List of detected change points.
        """
        permutations = np.array(permutations)
        T = len(permutations)

        # Filter outliers
        switch_rates = self.compute_switch_rate_sequence(permutations)
        valid_mask = switch_rates <= self.switch_threshold

        valid_indices = np.arange(T)[valid_mask]
        valid_permutations = permutations[valid_mask]

        if len(valid_permutations) < 10:
            return []  # Not enough valid data to detect change points

        # Change Point Detection using Binary Segmentation
        def calculate_segment_cost(segment_permutations):
            if len(segment_permutations) == 0: return 0
            if len(segment_permutations) < 2: return 0
            
            tuple_permutations = [tuple(p) for p in segment_permutations]
            mode_permutation = np.array(Counter(tuple_permutations).most_common(1)[0][0])

            cost = 0
            for perm in segment_permutations:
                cost += self.hamming_distance(perm, mode_permutation)
            return cost
        
        best_split = -1
        max_gain = -1

        total_cost = calculate_segment_cost(valid_permutations)

        # Search for single change point
        buffer = max(10, int(len(valid_permutations) * 0.1))

        for t in range(buffer, len(valid_permutations) - buffer):
            left_segment = valid_permutations[:t]
            right_segment = valid_permutations[t:]

            gain = total_cost - (calculate_segment_cost(left_segment) + calculate_segment_cost(right_segment))
            if gain > max_gain:
                max_gain = gain
                best_split = t
        
        # Significance Testing
        if best_split != -1:
            left_mode = Counter([tuple(p) for p in valid_permutations[:best_split]]).most_common(1)[0][0]
            right_mode = Counter([tuple(p) for p in valid_permutations[best_split:]]).most_common(1)[0][0]

            if left_mode == right_mode:
                return []  # No significant change point detected
            
            original_index = valid_indices[best_split]
            return [original_index]
        
        return []
    

def run_experiment():
    # Using synthetic data for demonstration
    np.random.seed(0)
    base_roles = np.arange(10)

    swapped_roles = base_roles.copy()
    swapped_roles[0], swapped_roles[9] = swapped_roles[9], swapped_roles[0]

    data = []

    for _ in range(100):
        # Add noise
        curr = base_roles.copy()
        if np.random.rand() < 0.1:
            i, j = np.random.choice(10, 2, replace=False)
            curr[i], curr[j] = curr[j], curr[i]
        data.append(curr)

    # Swap
    for _ in range(100):
        curr = swapped_roles.copy()
        if np.random.rand() < 0.1:
            i, j = np.random.choice(10, 2, replace=False)
            curr[i], curr[j] = curr[j], curr[i]
        data.append(curr)

    data = np.array(data)

    # Detector
    detector = RoleCPD()
    change_points = detector.fit_predict(data)

    print(f"True Change Point: 100")
    print(f"Detected Change Points: {change_points}")

    # Visualization
    plt.figure(figsize=(12, 6))

    # Distance matrix
    distances = [detector.hamming_distance(p, base_roles) for p in data]

    plt.plot(distances, label='Hamming Distance from Start', alpha=0.7)
    for cp in change_points:
        plt.axvline(cp, color='r', linestyle='--', label='Detected Change Points')
    
    plt.title('Role Change-Point Detection (Hamming Distance)')
    plt.xlabel('Frame')
    plt.ylabel('Hamming Distance (normalized)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_experiment()