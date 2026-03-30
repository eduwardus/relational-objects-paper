#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caracterización completa de átomos.
Salida: atom_database_unbiased.json
"""

import json
import numpy as np
from tqdm import tqdm

INPUT_FILE = "atoms_filtered_unbiased.json"
OUTPUT_FILE = "atom_database_unbiased.json"
g3 = 1.0

def characterize(adj):
    A = adj.astype(float)
    N = A.shape[0]
    
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(eigvals.real)
    lambda_star = eigvals[idx]
    vR = eigvecs[:, idx]
    
    eigvals_L, eigvecs_L = np.linalg.eig(A.T)
    idx_L = np.argmax(eigvals_L.real)
    vL = eigvecs_L[:, idx_L]
    
    vR = vR / np.linalg.norm(vR)
    vL = vL / np.linalg.norm(vL)
    
    ipr_R = np.sum(np.abs(vR)**4)
    ipr_L = np.sum(np.abs(vL)**4)
    
    real_vals = np.sort(eigvals.real)
    gap = real_vals[-1] - real_vals[-2] if len(real_vals) >= 2 else 0.0
    gap_rel = gap / lambda_star.real if lambda_star.real > 0 else 0.0
    
    vR_cubed = vR ** 3
    a3 = 3 * g3 * np.vdot(vL, vR_cubed)
    
    out_degrees = np.sum(A, axis=1)
    in_degrees = np.sum(A, axis=0)
    
    return {
        "sigma": float(lambda_star.real),
        "omega": float(lambda_star.imag),
        "Delta": float(gap),
        "gap_rel": float(gap_rel),
        "IPR_R": float(ipr_R),
        "IPR_L": float(ipr_L),
        "a3": float(a3.real),
        "out_degree_mean": float(np.mean(out_degrees)),
        "out_degree_std": float(np.std(out_degrees)),
        "in_degree_mean": float(np.mean(in_degrees)),
        "in_degree_std": float(np.std(in_degrees))
    }

def main():
    print("=== Caracterización de átomos ===")
    with open(INPUT_FILE, "r") as f:
        atoms = json.load(f)
    
    database = []
    for i, atom in enumerate(tqdm(atoms, desc="Caracterizando")):
        props = characterize(np.array(atom["adjacency_matrix"]))
        database.append({
            "id": i,
            "N": atom["N"],
            "p": atom["p"],
            "edge_density": atom["edge_density"],
            "lambda1": atom["lambda1"],
            "gap_rel_raw": atom["gap_rel"],
            "ipr_raw": atom["ipr"],
            **props
        })
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(database, f, indent=2)
    
    print(f"\n✅ Guardado en {OUTPUT_FILE}")
    print(f"📊 Total átomos: {len(database)}")

if __name__ == "__main__":
    main()
