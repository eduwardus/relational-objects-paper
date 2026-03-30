# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:11:29 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador no sesgado de grafos dirigidos (Erdős–Rényi).
Salida: graphs_raw_unbiased.json (todos los grafos fuertemente conexos)
->filter_atomic_unbiased.py
"""

import numpy as np
import json
import random
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from itertools import product

# ================= CONFIGURACIÓN =================
N_max = 20                    # N máximo
p_values = np.linspace(0.1, 0.9, 5)   # 5 densidades
NUM_SAMPLES_PER_P = 600        # muestras por (N, p)
OUTPUT_FILE = "graphs_raw_unbiased.json"
SEED = 42
# ================================================

random.seed(SEED)
np.random.seed(SEED)

def generate_random_directed(N, p):
    """Genera un grafo dirigido de Erdős–Rényi sin auto‑bucles."""
    adj = np.random.rand(N, N) < p
    np.fill_diagonal(adj, 0)   # eliminar diagonales
    return adj.astype(int)

def is_strongly_connected(adj):
    """Comprueba conectividad fuerte mediante componentes conexas."""
    n = adj.shape[0]
    if n == 1:
        return True
    sparse = csr_matrix(adj)
    n_comp, _ = connected_components(sparse, directed=True, connection='strong')
    return n_comp == 1

def main():
    all_graphs = []
    total_combinations = len(range(2, N_max + 1)) * len(p_values) * NUM_SAMPLES_PER_P

    print("=== Generador no sesgado (Erdős–Rényi) ===")
    print(f"N_max = {N_max}, p = {p_values}, muestras/p = {NUM_SAMPLES_PER_P}")

    with tqdm(total=total_combinations, desc="Generando grafos") as pbar:
        for N in range(2, N_max + 1):
            for p in p_values:
                accepted = 0
                for _ in range(NUM_SAMPLES_PER_P):
                    adj = generate_random_directed(N, p)
                    # Filtro mínimo: solo conectividad fuerte
                    if is_strongly_connected(adj):
                        all_graphs.append({
                            "N": N,
                            "p": float(p),
                            "adjacency_matrix": adj.tolist()
                        })
                        accepted += 1
                    pbar.update(1)
                # Logging de ratio de aceptación
                ratio = accepted / NUM_SAMPLES_PER_P
                tqdm.write(f"N={N}, p={p:.2f}: aceptados {accepted}/{NUM_SAMPLES_PER_P} ({ratio:.2%})")

    # Guardar en JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_graphs, f, indent=2)

    print(f"\nGuardado en {OUTPUT_FILE}")
    print(f"Total de grafos fuertemente conexos generados: {len(all_graphs)}")

if __name__ == "__main__":
    main()
