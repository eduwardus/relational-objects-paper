 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates the key figures for the paper from previously computed data.
Assumes the following files exist:
- atoms_filtered_unbiased.json
- observer_bootstrap_results.json
- observer_regions.json (optional, used to compute overlap histogram if not in bootstrap)
- (optional) markov_dynamics_relational.json to load π directly; otherwise recomputed.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from tqdm import tqdm
from itertools import combinations

# ================= CONFIGURATION =================
ATOMS_FILE = "atoms_filtered_unbiased.json"
BOOTSTRAP_FILE = "observer_bootstrap_results.json"
REGIONS_FILE = "observer_regions.json"
MARKOV_FILE = "markov_dynamics_relational.json"
OUTPUT_FIG1 = "fig_pca_stationary.png"
OUTPUT_FIG2 = "fig_overlap_hist.png"
OUTPUT_FIG3 = "fig_percolation.png"
OUTPUT_FIG4 = "fig_proto_objects.png"

# Parameters for recomputing π if needed
K_NEIGHBORS = 15
GAMMA = 0.3
EPSILON = 1e-4
TOP_K = 20
POWER_ITER = 500
# =================================================

def compute_triangles(adj):
    A = adj.astype(float)
    A3 = A @ A @ A
    return float(np.trace(A3).real)

def compute_base_features(atom):
    adj = np.array(atom["adjacency_matrix"])
    return [
        atom["N"],
        atom["edge_density"],
        atom["gap_rel"],
        atom["ipr"],
        compute_triangles(adj)
    ]

def build_transition_matrix(X, k_neighbors, gamma, epsilon):
    N = X.shape[0]
    D_geom = cdist(X, X)
    sigma = np.median(D_geom[D_geom > 0])
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    K_geom = np.zeros((N, N))
    for i in range(N):
        for j_idx, j in enumerate(indices[i]):
            d = distances[i, j_idx]
            K_geom[i, j] = np.exp(-d**2 / sigma**2)
        if indices[i][0] != i:
            K_geom[i, i] = 1.0
    D_obs = cdist(X, X)
    sigma_obs = np.median(D_obs[D_obs > 0])
    K_obs = np.exp(-D_obs**2 / sigma_obs**2)
    K = K_geom * K_obs
    row_sums = K.sum(axis=1, keepdims=True)
    P_base = K / (row_sums + 1e-12)
    P = (1 - gamma) * P_base + gamma * np.eye(N)
    P = (1 - epsilon) * P + epsilon / N
    return P, sigma

def stationary_distribution(P, max_iter=500, tol=1e-12):
    N = P.shape[0]
    pi = np.ones(N) / N
    for _ in range(max_iter):
        new_pi = pi @ P
        if np.linalg.norm(new_pi - pi) < tol:
            break
        pi = new_pi
    return pi / pi.sum()

def jaccard(set1, set2):
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0

def compute_overlaps_from_regions(regions_file):
    """Loads observer_regions.json and returns list of pairwise Jaccard overlaps."""
    with open(regions_file, "r") as f:
        regions_list = json.load(f)
    regions = [set(r) for r in regions_list]
    n_obs = len(regions)
    overlaps = []
    for i, j in combinations(range(n_obs), 2):
        overlaps.append(jaccard(regions[i], regions[j]))
    return np.array(overlaps)

def main():
    # 1. Load atoms and compute original features (N, density, gap_rel, ipr, triangles)
    with open(ATOMS_FILE, "r") as f:
        atoms = json.load(f)
    N = len(atoms)
    print(f"Loaded {N} atoms.")

    print("Computing original features (N, density, gap_rel, ipr, triangles)...")
    features = []
    for atom in tqdm(atoms):
        features.append(compute_base_features(atom))
    features = np.array(features)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)   # normalized features

    # 2. PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f"Variance explained: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

    # 3. Obtain stationary distribution π for the base observer (original features)
    pi = None
    try:
        with open(MARKOV_FILE, "r") as f:
            markov = json.load(f)
            if "stationary_distribution" in markov:
                pi = np.array(markov["stationary_distribution"])
                print("Loaded stationary distribution from markov_dynamics_relational.json")
    except:
        pass
    if pi is None:
        print("Recomputing stationary distribution for base observer...")
        P, _ = build_transition_matrix(X, K_NEIGHBORS, GAMMA, EPSILON)
        pi = stationary_distribution(P, max_iter=POWER_ITER)
        print("Computation finished.")

    # 4. Figure 1: PCA colored by π
    plt.figure(figsize=(6,5))
    sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=pi, s=10, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='π (stationary probability)')
    plt.title("Stationary distribution in PCA space")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG1, dpi=300)
    print(f"Figure 1 saved to {OUTPUT_FIG1}")

    # 5. Load bootstrap results and/or regions to obtain overlaps
    overlaps = None
    # First try to get overlaps from bootstrap JSON
    try:
        with open(BOOTSTRAP_FILE, "r") as f:
            bootstrap = json.load(f)
            if "overlap_stats" in bootstrap and "values" in bootstrap["overlap_stats"]:
                overlaps = np.array(bootstrap["overlap_stats"]["values"])
                print("Loaded overlaps from bootstrap results.")
    except:
        pass
    # If not found, try to compute from regions file
    if overlaps is None:
        try:
            overlaps = compute_overlaps_from_regions(REGIONS_FILE)
            print(f"Computed {len(overlaps)} overlaps from observer_regions.json.")
        except FileNotFoundError:
            print("Warning: observer_regions.json not found. Figure 2 will be skipped.")
    if overlaps is not None:
        # Figure 2: histogram of overlaps
        plt.figure(figsize=(6,4))
        plt.hist(overlaps, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel("Overlap (Jaccard) between stable regions")
        plt.ylabel("Frequency")
        plt.title("Distribution of overlap between observers")
        plt.axvline(np.mean(overlaps), color='red', linestyle='--', label=f'Mean = {np.mean(overlaps):.3f}')
        plt.axvline(np.median(overlaps), color='green', linestyle='--', label=f'Median = {np.median(overlaps):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG2, dpi=300)
        print(f"Figure 2 saved to {OUTPUT_FIG2}")

    # 6. Figure 3: percolation curve (components vs threshold)
    try:
        with open(BOOTSTRAP_FILE, "r") as f:
            bootstrap = json.load(f)
            families_by_threshold = bootstrap.get("families_by_threshold", {})
            if families_by_threshold:
                thresholds = sorted([float(t) for t in families_by_threshold.keys()])
                n_components = [families_by_threshold[str(t)]["n_components"] for t in thresholds]
                max_component = [families_by_threshold[str(t)]["max_size"] for t in thresholds]
                plt.figure(figsize=(6,4))
                plt.plot(thresholds, n_components, 'o-', label='Number of components')
                plt.plot(thresholds, max_component, 's-', label='Size of largest component')
                plt.xlabel("Overlap threshold")
                plt.ylabel("Count / size")
                plt.legend()
                plt.title("Structure of observer space")
                plt.tight_layout()
                plt.savefig(OUTPUT_FIG3, dpi=300)
                print(f"Figure 3 saved to {OUTPUT_FIG3}")
            else:
                print("Warning: families_by_threshold data not found. Figure 3 skipped.")
    except FileNotFoundError:
        print("Warning: observer_bootstrap_results.json not found. Figure 3 skipped.")

    # 7. Figure 4: Stable region (top-20) of the base observer in PCA
    top_ids = np.argsort(pi)[-TOP_K:][::-1]   # top 20 states
    plt.figure(figsize=(6,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c='lightgray', s=5, alpha=0.5, label='Other states')
    plt.scatter(X_pca[top_ids,0], X_pca[top_ids,1],
                c='red', s=20, edgecolors='black', label='Stable region (top 20)')
    plt.legend()
    plt.title("Emergence of proto‑objects (base observer)")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG4, dpi=300)
    print(f"Figure 4 saved to {OUTPUT_FIG4}")

    print("\n✅ All figures generated.")

if __name__ == "__main__":
    main()
