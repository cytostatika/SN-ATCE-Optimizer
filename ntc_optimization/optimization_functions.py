from numpy.typing import NDArray
import numpy as np
import cvxpy as cp


def ntc_optimization_function(ntc_value_vector: NDArray, normalization=1) -> float:
    pairwise_matrix = ntc_value_vector.reshape((ntc_value_vector.shape[0] // 2, 2))
    pairwise_summed_vector = np.log1p(pairwise_matrix.sum(axis=1) / normalization)
    return -np.sum(pairwise_summed_vector)


def ram_constraint(ntc_value_vector: NDArray, ram_value: float, ptdfs: NDArray, epsilon: int = 1):
    return epsilon + ram_value - np.dot(ntc_value_vector, ptdfs)


def leq_lower_bound_constraint(ntc_value_vector, aac_value: float, index: int, epsilon: int = 1):
    return ntc_value_vector[index] - (-epsilon + aac_value)


def geq_lower_bound_constraint(ntc_value_vector, aac_value: float, index: int, epsilon: int = 1):
    return epsilon + aac_value - ntc_value_vector[index]
