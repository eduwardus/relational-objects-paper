# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:49:28 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dinámica Markoviana con proyección observacional y memoria.
- Features: N, density, gap_rel, ipr, triangles.
- Proyección: [density, gap_rel] (normalizados).
- Kernel geométrico: KNN (k=50) + kernel gaussiano.
- Kernel observacional: global (todas las distancias) + kernel gaussiano.
- Combinación: producto elemento a elemento.
- Memoria: refuerzo diagonal (GAMMA).
- Ergodicidad: ruido uniforme (EPSILON).
- Salida: distribución estacionaria, entropía, gap espectral.
"""

import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ================= CONFIGURACIÓN =================
INPUT_FILE = "atoms_filtered_unbiased.json"   # contiene matrices y propiedades básicas
OUTPUT_FILE = "markov_obs_memory.json"

K_NEIGHBORS = 50
GAMMA = 0.3          # peso de la memoria (diagonal)
EPSILON = 1e-4       # ruido ergódico
SIGMA_SCALE = 1.0
SIGMA_OBS_SCALE = 1.0
# ================================================

def compute_triangles(adj):
    """Número de ciclos dirigidos de longitud 3 (triángulos)."""
    A = adj.astype(float)
    # Para grafos dirigidos, la traza de A^3 cuenta ciclos dirigidos de 3 nodos.
    # Se utiliza como proxy de la densidad de ciclos.
    A3 = A @ A @ A
    return float(np.trace(A3).real)

def build_transition_matrix(features, O):
    """
    features: array (N, d) normalizada
    O: array (N, p) observación normalizada (subconjunto de features)
    """
    N = features.shape[0]

    # --- Kernel geométrico (KNN) ---
    D_geom = cdist(features, features)
    sigma_geom = np.median(D_geom) * SIGMA_SCALE
    print(f"Sigma geométrico = {sigma_geom:.6f}")

    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='euclidean').fit(features)
    distances, indices = nbrs.kneighbors(features)

    K_geom = np.zeros((N, N))
    for i in range(N):
        for j_idx, j in enumerate(indices[i]):
            d = distances[i, j_idx]
            K_geom[i, j] = np.exp(-d**2 / sigma_geom**2)
        K_geom[i, i] = 1.0   # auto‑loop

    # --- Kernel observacional (global) ---
    D_obs = cdist(O, O)
    sigma_obs = np.median(D_obs) * SIGMA_OBS_SCALE
    print(f"Sigma observacional = {sigma_obs:.6f}")
    K_obs = np.exp(-D_obs**2 / sigma_obs**2)

    # --- Kernel combinado ---
    K = K_geom * K_obs

    # --- Matriz de transición base ---
    row_sums = K.sum(axis=1, keepdims=True)
    P_base = K / row_sums

    # --- Memoria: refuerzo diagonal ---
    P = (1 - GAMMA) * P_base + GAMMA * np.eye(N)

    # --- Ergodicidad: ruido uniforme ---
    P = (1 - EPSILON) * P + EPSILON / N

    return P, sigma_geom, sigma_obs

def stationary_distribution(P, tol=1e-12, max_iter=10000):
    N = P.shape[0]
    pi = np.ones(N) / N
    for _ in range(max_iter):
        new_pi = pi @ P
        if np.linalg.norm(new_pi - pi) < tol:
            break
        pi = new_pi
    return pi / pi.sum()

def spectral_gap(P):
    """Calcula el gap espectral (λ1 - λ2) usando los dos autovalores de mayor parte real."""
    try:
        # Usamos eigs para matrices grandes
        from scipy.sparse.linalg import eigs
        vals = eigs(P, k=2, which='LM', return_eigenvectors=False)
        vals = np.real(vals)
        vals = np.sort(vals)[::-1]
        gap = vals[0] - vals[1]
        mixing_time = 1.0 / (gap + 1e-12)
        return gap, mixing_time
    except Exception as e:
        print(f"Error en espectro: {e}")
        return 0.0, np.inf

def main():
    print("=== Dinámica Markoviana con observación y memoria ===")
    with open(INPUT_FILE, "r") as f:
        atoms = json.load(f)
    N = len(atoms)
    print(f"Total átomos: {N}")

    # --- Extraer features ---
    features_raw = []
    for atom in tqdm(atoms, desc="Extrayendo features"):
        adj = np.array(atom["adjacency_matrix"])
        N_val = atom["N"]
        density = atom["edge_density"]
        gap_rel = atom["gap_rel"]
        ipr = atom["ipr"]
        triangles = compute_triangles(adj)
        features_raw.append([N_val, density, gap_rel, ipr, triangles])

    features_raw = np.array(features_raw)

    # --- Normalizar features ---
    scaler = StandardScaler()
    X = scaler.fit_transform(features_raw)   # shape (N, 5)

    # --- Observación: solo densidad y gap_rel (índices 1 y 2) ---
    # Nota: después de normalizar, los índices corresponden a la misma posición
    O = X[:, [1, 2]]   # density, gap_rel

    # --- Construir matriz de transición ---
    P, sigma_geom, sigma_obs = build_transition_matrix(X, O)

    # --- Distribución estacionaria ---
    pi = stationary_distribution(P)

    # --- Métricas ---
    entropy = -np.sum(pi * np.log(pi + 1e-12))
    max_pi = np.max(pi)
    min_pi = np.min(pi)
    std_pi = np.std(pi)

    gap, mixing_time = spectral_gap(P)

    # --- Top estados ---
    top_idx = np.argsort(pi)[-10:][::-1]

    print("\n" + "="*50)
    print("📊 RESULTADOS")
    print("="*50)
    print(f"Entropía: {entropy:.4f} (máx: {np.log(N):.4f})")
    print(f"π: max={max_pi:.6f}, min={min_pi:.6f}, std={std_pi:.6f}")
    print(f"Gap espectral: {gap:.6f}")
    print(f"Tiempo de mezcla estimado: {mixing_time:.1f} pasos")

    print("\nTop 10 estados más probables:")
    for idx in top_idx:
        atom = atoms[idx]
        print(f"  id={atom.get('id', idx):3d}, N={atom['N']:2d}, density={atom['edge_density']:.3f}, "
              f"gap={atom['gap_rel']:.4f}, ipr={atom['ipr']:.4f}, π={pi[idx]:.6f}")

    # --- Guardar resultados ---
    results = {
        "description": "Markov dynamics with observation projection and memory",
        "parameters": {
            "k_neighbors": K_NEIGHBORS,
            "gamma": GAMMA,
            "epsilon": EPSILON,
            "sigma_scale": SIGMA_SCALE,
            "sigma_obs_scale": SIGMA_OBS_SCALE
        },
        "sigma_geom": sigma_geom,
        "sigma_obs": sigma_obs,
        "n_states": N,
        "stationary_distribution": pi.tolist(),
        "entropy": entropy,
        "spectral_gap": gap,
        "mixing_time": mixing_time,
        "top_states": [
            {
                "index": int(idx),
                "N": atoms[idx]["N"],
                "edge_density": atoms[idx]["edge_density"],
                "gap_rel": atoms[idx]["gap_rel"],
                "ipr": atoms[idx]["ipr"],
                "pi": pi[idx]
            }
            for idx in top_idx
        ]
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Guardado en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
