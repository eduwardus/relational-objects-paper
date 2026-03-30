#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtro atómico estratificado por N.
Selecciona top 5% de gap_rel dentro de cada N.
Elimina duplicados por firma estructural.
->build_proto_objects.py
"""

import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ================= CONFIGURACIÓN =================
INPUT_FILE = "graphs_raw_unbiased.json"
OUTPUT_FILE = "atoms_filtered_unbiased.json"
GAP_PERCENTILE_PER_N = 95   # top 5% por gap relativo dentro de cada N
# ================================================

def compute_gap_and_ipr(adj):
    """
    Calcula gap relativo e IPR de forma robusta.
    Retorna (lambda1, gap_rel, ipr) o (None, None, None) si inválido.
    """
    A = adj.astype(float)
    eigvals, eigvecs = np.linalg.eig(A)
    
    # Autovalor dominante (mayor parte real)
    idx = np.argmax(eigvals.real)
    lambda1 = eigvals[idx].real
    
    if lambda1 <= 0:
        return None, None, None
    
    # Gap relativo (necesita segundo autovalor real)
    real_vals = eigvals.real
    if len(real_vals) >= 2:
        sorted_real = np.sort(real_vals)
        lambda2 = sorted_real[-2]
        gap_rel = (lambda1 - lambda2) / lambda1
    else:
        gap_rel = 0.0
    
    # IPR del autovector dominante
    v = eigvecs[:, idx].real
    v = v / np.linalg.norm(v)
    ipr = np.sum(v**4)
    
    return lambda1, gap_rel, ipr


def get_graph_signature(adj, N):
    """
    Firma única para evitar duplicados estructurales.
    Combina: tamaño, espectro redondeado, secuencia de grados.
    """
    # Espectro redondeado (tolera pequeñas diferencias numéricas)
    eigvals = np.linalg.eigvals(adj.astype(float))
    rounded = np.round(eigvals, 6)
    spectral_key = tuple(sorted([(val.real, val.imag) for val in rounded]))
    
    # Secuencias de grados
    out_degrees = tuple(sorted(np.sum(adj, axis=1)))
    in_degrees = tuple(sorted(np.sum(adj, axis=0)))
    
    return (N, spectral_key, out_degrees, in_degrees)


def main():
    print("=== Filtro atómico estratificado por N ===")
    print(f"Percentil por N: {GAP_PERCENTILE_PER_N}%")
    
    with open(INPUT_FILE, "r") as f:
        graphs = json.load(f)
    
    print(f"Total de grafos en entrada: {len(graphs)}")
    
    # Agrupar por N
    by_N = defaultdict(list)
    for g in graphs:
        by_N[g["N"]].append(g)
    
    all_atoms = []
    
    for N in sorted(by_N.keys()):
        print(f"\n📊 Procesando N={N} ({len(by_N[N])} grafos)")
        
        # Calcular propiedades para este N
        props = []
        for g in tqdm(by_N[N], desc=f"N={N}", leave=False):
            adj = np.array(g["adjacency_matrix"])
            lambda1, gap_rel, ipr = compute_gap_and_ipr(adj)
            if lambda1 is not None:
                props.append({
                    "graph": g,
                    "adj": adj,
                    "lambda1": lambda1,
                    "gap_rel": gap_rel,
                    "ipr": ipr
                })
        
        if not props:
            print(f"  ⚠️ Sin grafos válidos para N={N}")
            continue
        
        # Umbral dentro de este N
        gaps = np.array([p["gap_rel"] for p in props])
        threshold = np.percentile(gaps, GAP_PERCENTILE_PER_N)
        
        print(f"  Umbral gap_rel (P{GAP_PERCENTILE_PER_N}): {threshold:.6f}")
        print(f"  Rango gaps: min={np.min(gaps):.6f}, max={np.max(gaps):.6f}")
        
        # Seleccionar top percentil y eliminar duplicados
        seen = set()
        selected = []
        for p in props:
            if p["gap_rel"] >= threshold:
                g = p["graph"]
                adj = p["adj"]
                signature = get_graph_signature(adj, N)
                if signature not in seen:
                    seen.add(signature)
                    selected.append({
                        "N": N,
                        "p": g["p"],
                        "adjacency_matrix": adj.tolist(),
                        "edge_density": float(np.sum(adj) / (N * (N - 1))),
                        "lambda1": float(p["lambda1"]),
                        "gap_rel": float(p["gap_rel"]),
                        "ipr": float(p["ipr"])
                    })
        
        print(f"  ✅ Seleccionados: {len(selected)}/{len(props)}")
        all_atoms.extend(selected)
    
    # Guardar resultados
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_atoms, f, indent=2)
    
    # Estadísticas finales
    print(f"\n{'='*50}")
    print(f"✅ TOTAL ÁTOMOS: {len(all_atoms)}")
    print(f"📁 Guardado en: {OUTPUT_FILE}")
    
    from collections import Counter
    n_counts = Counter([a["N"] for a in all_atoms])
    print(f"\n📈 Distribución por N:")
    for N in sorted(n_counts.keys()):
        bar = "█" * int(50 * n_counts[N] / max(n_counts.values()))
        print(f"  N={N:2d}: {n_counts[N]:3d} átomos {bar}")
    
    # Estadísticas de propiedades
    iprs = [a["ipr"] for a in all_atoms]
    gaps = [a["gap_rel"] for a in all_atoms]
    print(f"\n📊 Propiedades globales:")
    print(f"  IPR:   media={np.mean(iprs):.4f}, std={np.std(iprs):.4f}")
    print(f"  gap_rel: media={np.mean(gaps):.4f}, std={np.std(gaps):.4f}")


if __name__ == "__main__":
    main()
