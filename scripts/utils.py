#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for the relational objects project.
"""

import numpy as np
import networkx as nx

def compute_triangles(adj):
    """Number of directed triangles (trace of A^3)."""
    A = adj.astype(float)
    A3 = A @ A @ A
    return float(np.trace(A3).real)

def compute_spectral_entropy(adj):
    """Shannon entropy of eigenvalue magnitudes."""
    eigvals = np.linalg.eigvals(adj.astype(float))
    mag = np.abs(eigvals)
    mag = mag / (mag.sum() + 1e-12)
    return -np.sum(mag * np.log(mag + 1e-12))

def compute_clustering_coefficient(adj):
    """Mean clustering coefficient for directed graphs."""
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    try:
        return np.mean(list(nx.clustering(G).values()))
    except:
        return 0.0

def compute_algebraic_connectivity(adj):
    """Second smallest eigenvalue of the Laplacian (undirected underlying)."""
    undirected = adj + adj.T
    undirected[undirected > 0] = 1
    deg = np.sum(undirected, axis=1)
    L = np.diag(deg) - undirected
    eigvals = np.linalg.eigvals(L)
    eigvals = np.sort(eigvals.real)
    if len(eigvals) >= 2:
        return eigvals[1]
    return 0.0

def compute_betweenness_mean(adj):
    """Mean betweenness centrality for directed graphs."""
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    try:
        bc = list(nx.betweenness_centrality(G).values())
        return float(np.mean(bc))
    except:
        return 0.0

def compute_base_features(atom):
    """Extract original features: N, density, gap_rel, ipr, triangles."""
    adj = np.array(atom["adjacency_matrix"])
    return [
        atom["N"],
        atom["edge_density"],
        atom["gap_rel"],
        atom["ipr"],
        compute_triangles(adj)
    ]

def compute_topological_features(atom):
    """Extract topological features: entropy, clustering, connectivity, betweenness."""
    adj = np.array(atom["adjacency_matrix"])
    return [
        compute_spectral_entropy(adj),
        compute_clustering_coefficient(adj),
        compute_algebraic_connectivity(adj),
        compute_betweenness_mean(adj)
    ]
