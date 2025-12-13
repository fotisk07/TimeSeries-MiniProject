import numpy as np
from soccercpd.record_manager import RecordManager
from soccercpd.match import Match

rng = np.random.default_rng(0)

def simulate_roles(T=400, cp=200, sigma=2.0):
    """
    Simulates role trajectories with a single formation change.
    """
    F1 = np.array([
    (-40,  20), (-40,   7), (-40,  -7), (-40, -20),   # Defenders (4)
    (-10,  20), (-10,   7), (-10,  -7), (-10, -20),   # Midfield (4)
    ( 25,   8), ( 25,  -8)                            # Forwards (2)
    ])
    F2 = np.array([
        (-40,  10), (-40,   0), (-40, -10),               # Defenders (3)
        (-10,  20), (-10,   7), (-10,  -7), (-10, -20),   # Midfield (4)
        ( 25,  18), ( 25,   0), ( 25, -18)                # Forwards (3)
    ])
    N = len(F1)
    V = np.zeros((T, N, 2))

    for t in range(T):
        mu = F1 if t < cp else F2
        V[t] = mu + sigma * np.random.randn(N,2)

    return V

def simulate_swap(T=300, sigma=0.5):
    # 6-role formation
    mu = np.array([
        [-8,  2], [-2, 2], [4, 2],
        [-4, -2], [2, -2],
        [0, -6]
    ])

    N = len(mu)
    X = np.zeros((T, N, 2))

    # players assigned fixed roles initially
    roles = np.arange(N)

    # choose two roles to swap
    k1, k2 = 3, 4

    for t in range(T):
        alpha = 0
        if T//3 <= t <= 2*T//3:
            alpha = (t - T//3) / (T//3)

        for i in range(N):
            k = roles[i]
            center = mu[k]

            # interpolate swap
            if k == k1:
                center = (1-alpha)*mu[k1] + alpha*mu[k2]
            if k == k2:
                center = (1-alpha)*mu[k2] + alpha*mu[k1]

            X[t, i] = center + sigma * rng.standard_normal(2)

    return X, mu

def generate_synthetic_role_data(T=450, N=10, switch_times=[150, 300], noise_level=0.1):
    """
    Generates a sequence of role permutations.
    Returns: Array of shape (T, N) where row t is the permutation at time t.
    """
    # Phase 1
    # Base permutation: Identity
    base_roles = np.arange(N)
    
    # Phase 2
    # Tactical Change: Swap roles 7 and 9
    phase2_roles = base_roles.copy()
    phase2_roles[7], phase2_roles[9] = 9, 7

    # Phase 3
    # Tactical Change: Swap roles 3 and 4
    phase3_roles = base_roles.copy()
    phase3_roles[3], phase3_roles[4] = 4, 3

    permutations = []
    
    for t in range(T):
        # Determine dominant tactic
        if t < switch_times[0]:
            current_perm = base_roles.copy()
        elif t < switch_times[1]:
            current_perm = phase2_roles.copy()
        else:
            current_perm = phase3_roles.copy()
            
        # Add temporary noise (random temporary swaps)
        if np.random.rand() < noise_level:
            # Randomly swap two players to simulate temporary cover/overlap
            i, j = np.random.choice(N, 2, replace=False)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            
        permutations.append(current_perm)
        
    return np.array(permutations)



def load_and_prepare_match(activity_id=17985):
    rm = RecordManager()
    activity_record, player_periods, roster, ugp = rm.load_activity_data(activity_id)

    match = Match(activity_record, player_periods, roster, ugp)
    match.construct_inplay_ugp()
    match.rotate_pitch()

    return match
