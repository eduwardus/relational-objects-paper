# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:29:57 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap sobre el espacio de observadores: genera múltiples proyecciones aleatorias
de un pool de features, construye la dinámica Markoviana, extrae la región estable
(top-K) para cada observador y calcula el solapamiento entre regiones.
Además, analiza la estructura de familias para diferentes umbrales de solapamiento
(umbrales fijos y percentiles de la distribución).
Guarda las regiones en observer_regions.json y los resultados en observer_bootstrap_results.json.
"""

import json
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# ================= CONFIGURACIÓN =================
INPUT_ATOMS = "atoms_filtered_unbiased.json"
OUTPUT_JSON = "observer_bootstrap_results.json"
OUTPUT_PLOT = "observer_bootstrap_overlap_hist.png"
OUTPUT_REGIONS = "observer_regions.json"

N_OBSERVERS = 300
SUBSPACE_DIM = random.choice([2,3,4])
K_NEIGHBORS = 20
GAMMA = random.choice([0.2, 0.3, 0.4])
EPSILON = 1e-4
TOP_K = 30
POWER_ITER = 800
# Umbrales fijos para análisis de familias
FIXED_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
# Percentiles de la distribución de solapamientos a usar como umbrales
PERCENTILES = [50, 75, 90, 95, 99]
# =================================================

# --------------------------------------------------
# Funciones auxiliares para calcular features base
# --------------------------------------------------

def compute_triangles(adj):
    A = adj.astype(float)
    A3 = A @ A @ A
    return float(np.trace(A3).real)

def compute_spectral_entropy(adj):
    eigvals = np.linalg.eigvals(adj.astype(float))
    mag = np.abs(eigvals)
    mag = mag / (mag.sum() + 1e-12)
    return -np.sum(mag * np.log(mag + 1e-12))

def compute_clustering_coefficient(adj):
    import networkx as nx
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    try:
        return np.mean(list(nx.clustering(G).values()))
    except:
        return 0.0

def compute_algebraic_connectivity(adj):
    # grafo no dirigido subyacente
    undirected = adj + adj.T
    undirected[undirected > 0] = 1
    deg = np.sum(undirected, axis=1)
    L = np.diag(deg) - undirected
    eigvals = np.linalg.eigvals(L)
    eigvals = np.sort(eigvals.real)
    if len(eigvals) >= 2:
        return eigvals[1]
    else:
        return 0.0

def compute_betweenness_mean(adj):
    import networkx as nx
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    try:
        bc = list(nx.betweenness_centrality(G).values())
        return float(np.mean(bc))
    except:
        return 0.0

def compute_base_features(atom):
    """Calcula el vector de 8 features base."""
    adj = np.array(atom["adjacency_matrix"])
    return [
        atom["edge_density"],                 # 0 density
        atom["gap_rel"],                      # 1 gap_rel
        atom["ipr"],                          # 2 ipr
        compute_triangles(adj),               # 3 triangles
        compute_spectral_entropy(adj),        # 4 entropy
        compute_clustering_coefficient(adj),  # 5 clustering
        compute_algebraic_connectivity(adj),  # 6 algebraic_conn
        compute_betweenness_mean(adj)         # 7 betweenness
    ]

# --------------------------------------------------
# Funciones de construcción de dinámica
# --------------------------------------------------

def build_transition_matrix(X, k_neighbors, gamma, epsilon):
    """
    X: matriz (N, d) normalizada
    Retorna matriz estocástica P (N x N) y sigma geométrico.
    """
    N = X.shape[0]
    # Distancias geométricas
    D_geom = cdist(X, X)
    sigma = np.median(D_geom[D_geom > 0])  # evitar diagonales cero
    # Kernel geométrico (KNN)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    K_geom = np.zeros((N, N))
    for i in range(N):
        for j_idx, j in enumerate(indices[i]):
            d = distances[i, j_idx]
            K_geom[i, j] = np.exp(-d**2 / sigma**2)
        # asegurar auto‑bucle (distancia 0 → 1) si no está incluido
        if indices[i][0] != i:
            K_geom[i, i] = 1.0
    # Kernel observacional = distancia global (en el mismo espacio)
    D_obs = cdist(X, X)
    sigma_obs = np.median(D_obs[D_obs > 0])
    K_obs = np.exp(-D_obs**2 / sigma_obs**2)
    # Kernel combinado
    K = K_geom * K_obs
    # Normalización base
    row_sums = K.sum(axis=1, keepdims=True)
    P_base = K / (row_sums + 1e-12)
    # Memoria
    P = (1 - gamma) * P_base + gamma * np.eye(N)
    # Ergodicidad
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

def top_k(pi, k=20):
    return [int(i) for i in np.argsort(pi)[-k:][::-1]]

# --------------------------------------------------
# Funciones de generación de observadores
# --------------------------------------------------

def random_subset_observer(base_matrix, dim):
    """Elige aleatoriamente 'dim' columnas de la matriz base."""
    n_features = base_matrix.shape[1]
    idx = random.sample(range(n_features), dim)
    return base_matrix[:, idx]

def random_projection_observer(base_matrix, dim):
    """Proyección lineal aleatoria: base_matrix @ W, W ~ N(0,1)."""
    n_features = base_matrix.shape[1]
    W = np.random.randn(n_features, dim)
    return base_matrix @ W

def generate_observer(base_matrix, dim):
    """Elige aleatoriamente el tipo de observador (con 50% de probabilidad cada uno)."""
    if random.random() < 0.5:
        return random_subset_observer(base_matrix, dim)
    else:
        return random_projection_observer(base_matrix, dim)

# --------------------------------------------------
# Análisis de familias para múltiples umbrales
# --------------------------------------------------

def compute_components_for_threshold(overlap_matrix, thresh):
    """Dada una matriz de solapamientos (simétrica), construye el grafo y devuelve componentes."""
    n = overlap_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if overlap_matrix[i, j] > thresh:
                G.add_edge(i, j)
    components = list(nx.connected_components(G))
    sizes = [len(c) for c in components]
    return {
        "n_components": len(components),
        "max_size": max(sizes) if sizes else 0,
        "mean_size": float(np.mean(sizes)) if sizes else 0,
        "median_size": float(np.median(sizes)) if sizes else 0,
        "size_distribution": {int(k): v for k, v in dict(nx.degree(G)).items()}
    }

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("=== Bootstrap sobre el espacio de observadores ===")
    # Cargar átomos
    with open(INPUT_ATOMS, "r") as f:
        atoms = json.load(f)
    N = len(atoms)
    print(f"Cargados {N} átomos.")

    # Calcular features base para todos los átomos
    print("Calculando features base (8 dimensiones)...")
    base_features = []
    for atom in tqdm(atoms):
        base_features.append(compute_base_features(atom))
    base_features = np.array(base_features)   # shape (N, 8)

    # Normalizar base (para que las proyecciones partan de datos normalizados)
    scaler_base = StandardScaler()
    base_norm = scaler_base.fit_transform(base_features)

    # Almacenar regiones estables para cada observador
    regions = []  # lista de conjuntos de IDs

    print(f"\nGenerando {N_OBSERVERS} observadores aleatorios...")
    for obs_idx in tqdm(range(N_OBSERVERS), desc="Observadores"):
        # Generar features proyectadas
        X_proj = generate_observer(base_norm, SUBSPACE_DIM)   # shape (N, dim)
        # Normalizar nuevamente (para el kernel)
        scaler = StandardScaler()
        X = scaler.fit_transform(X_proj)

        # Construir matriz de transición
        P, _ = build_transition_matrix(X, K_NEIGHBORS, GAMMA, EPSILON)

        # Distribución estacionaria
        pi = stationary_distribution(P, max_iter=POWER_ITER)

        # Región estable: top-K
        region = set(top_k(pi, k=TOP_K))
        regions.append(region)

    # Precalcular matriz de solapamientos (solo triangular superior)
    n_obs = len(regions)
    print("\nCalculando matriz de solapamientos...")
    overlap_matrix = np.zeros((n_obs, n_obs))
    for i in tqdm(range(n_obs), desc="Pares i"):
        for j in range(i+1, n_obs):
            inter = len(regions[i] & regions[j])
            union = len(regions[i] | regions[j])
            jac = inter / union if union > 0 else 0.0
            overlap_matrix[i, j] = jac
            overlap_matrix[j, i] = jac

    # Obtener todos los valores de solapamiento (sin diagonal)
    all_overlaps = overlap_matrix[np.triu_indices(n_obs, k=1)]
    mean_overlap = np.mean(all_overlaps)
    median_overlap = np.median(all_overlaps)
    zero_fraction = np.mean(all_overlaps < 0.05)
    max_overlap = np.max(all_overlaps)

    # Calcular percentiles de la distribución
    percentiles_vals = {p: np.percentile(all_overlaps, p) for p in PERCENTILES}

    # Umbrales a evaluar: fijos + percentiles
    thresholds = list(FIXED_THRESHOLDS) + list(percentiles_vals.values())
    thresholds = sorted(set(thresholds))

    # Analizar componentes para cada umbral
    families_by_threshold = {}
    for thresh in thresholds:
        comp_stats = compute_components_for_threshold(overlap_matrix, thresh)
        families_by_threshold[str(thresh)] = comp_stats

    # Guardar regiones para análisis posterior
    regions_list = [list(r) for r in regions]
    with open(OUTPUT_REGIONS, "w") as f:
        json.dump(regions_list, f)
    print(f"Regiones guardadas en {OUTPUT_REGIONS}")

    # Guardar resultados estadísticos en JSON
    results = {
        "n_observers": N_OBSERVERS,
        "subspace_dim": SUBSPACE_DIM,
        "k_neighbors": K_NEIGHBORS,
        "gamma": GAMMA,
        "epsilon": EPSILON,
        "top_k": TOP_K,
        "power_iterations": POWER_ITER,
        "overlap_stats": {
            "mean": float(mean_overlap),
            "median": float(median_overlap),
            "zero_fraction": float(zero_fraction),
            "max": float(max_overlap),
            "percentiles": {p: float(percentiles_vals[p]) for p in PERCENTILES}
        },
        "families_by_threshold": families_by_threshold
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en {OUTPUT_JSON}")

    # Histograma de solapamientos
    plt.figure(figsize=(8,5))
    plt.hist(all_overlaps, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(mean_overlap, color='red', linestyle='--', label=f'Mean = {mean_overlap:.3f}')
    plt.axvline(median_overlap, color='green', linestyle='--', label=f'Median = {median_overlap:.3f}')
    plt.xlabel('Jaccard overlap')
    plt.ylabel('Frequency')
    plt.title('Distribución de solapamientos entre observadores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"Histograma guardado en {OUTPUT_PLOT}")

    # Mostrar tabla de evolución de familias
    print("\n=== Evolución de familias con el umbral ===")
    print("Umbral | Comp. | Max. comp. | Media | Mediana")
    print("-" * 50)
    for thresh in thresholds:
        comp = families_by_threshold[str(thresh)]
        print(f"{thresh:6.3f} | {comp['n_components']:5d} | {comp['max_size']:10d} | {comp['mean_size']:5.1f} | {comp['median_size']:7.1f}")

if __name__ == "__main__":
    main()
