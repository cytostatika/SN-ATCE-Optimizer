# %%
from itertools import combinations
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp
from numpy.typing import NDArray

import logging
from ntc_optimization.dataclasses_and_config import (
    ConstraintOptions,
    FlowSituationforMTU,
    NTCSolutionForMTU,
    OptimizationConfig,
    OptimizationParameters,
)


logging.basicConfig(encoding="utf-8", level=logging.INFO)

SOLVER = cp.CLARABEL
SOLVER_SETTINGS = dict(
    tol_feas=1e-4, tol_gap_abs=1e-4, tol_gap_rel=1e-4, tol_infeas_abs=5e-5, tol_infeas_rel=5e-5, tol_ktratio=1e-4
)


def make_constraints(
    constraint_options: ConstraintOptions,
    ntc_value_vector: cp.Variable,
    ram_values: NDArray,
    ptdfs: NDArray,
    aac: NDArray,
    hvdc_index_pairs: list[tuple[int, int]],
    hvdc_ilf: list[float],
    non_id_corridor_indices: list[int],
    corridor_names: list[str],
    cnec_names: list[str],
):
    # Constraints setup
    constraints: list[cp.Constraint] = []
    constraint_names: list[str] = []
    include_max_ntc_se2_no4 = False
    if include_max_ntc_se2_no4:
        # Add constraint for max SE2 to NO4 constraint
        constraints_se2_no4 = []
        constraints_se2_no4_names = []
        max_ntc_se2_no4 = 500
        for i, pair in enumerate(corridor_names):
            pair = pair.replace("z2z_", "").split('-')
            if pair[0] == 'SE2' and pair[1] == 'NO4':
                constraints_se2_no4.append(ntc_value_vector[i] <= max_ntc_se2_no4)
                constraints_se2_no4_names.append('Max NTC SE2 > NO4')
        constraints += constraints_se2_no4
        constraint_names += constraints_se2_no4_names

    #constraints: list[cp.Constraint] = []
    #constraint_names: list[str] = []
    include_max_ntc_no5_no2 = False
    if include_max_ntc_no5_no2:
        # Add constraint for max NO5 to NO2 constraint
        constraints_no5_no2 = []
        constraints_no5_no2_names = []
        max_ntc_no5_no2 = 1000
        for i, pair in enumerate(corridor_names):
            pair = pair.replace("z2z_", "").split('-')
            if pair[0] == 'NO5' and pair[1] == 'NO2':
                constraints_no5_no2.append(ntc_value_vector[i] <= max_ntc_no5_no2)
                constraints_no5_no2_names.append('Max NTC NO5 > NO2')
        constraints += constraints_no5_no2
        constraint_names += constraints_no5_no2_names

    include_max_ntc_no2_no5 = False
    if include_max_ntc_no2_no5:
        # Add constraint for max NO5 to NO2 constraint
        constraints_no2_no5 = []
        constraints_no2_no5_names = []
        max_ntc_no2_no5 = 1000
        for i, pair in enumerate(corridor_names):
            pair = pair.replace("z2z_", "").split('-')
            if pair[0] == 'NO2' and pair[1] == 'NO5':
                constraints_no2_no5.append(ntc_value_vector[i] <= max_ntc_no2_no5)
                constraints_no2_no5_names.append('Max NTC NO2 > NO5')
        constraints += constraints_no2_no5
        constraint_names += constraints_no2_no5_names

    

    # NOTE: this for loop adds RAM constraints for each row in ptdfs matrix
    epsilon = 1
    if constraint_options.ram_constraints:
        flow_less_than_ram_constraints, ram_constraint_names = make_ram_constraints(
            ntc_value_vector, ram_values, ptdfs, cnec_names, epsilon
        )
        constraints += flow_less_than_ram_constraints
        constraint_names += ram_constraint_names

    # NOTE: this for loop adds AAC constraints. This constraint enforces that we give at least the AAC to the ID market.
    # We handle both less than and greater than cases based on AAC values by sign-flipping the terms in the constraint

    # BUG: Ensure that at least one side, or maybe both, of the Nordic HVDC has import-export constraints!!!!
    if constraint_options.minimum_aac:
        ntc_atleast_aac_constraints, aac_constraint_names = make_ntc_atleast_aac_constraints(
            ntc_value_vector, aac, corridor_names, epsilon
        )
        constraints += ntc_atleast_aac_constraints
        constraint_names += aac_constraint_names

    # NOTE: this for loop sets the ID capacity to ~0 for corridors that do not participate in the market
    # the limit is imposed by adding an upper bound on the caacity equal to the AAC + epsilon
    if constraint_options.maximum_aac_for_non_id:
        max_aac_constraints, max_aac_constraint_names = make_ntc_max_aac_constraints(
            ntc_value_vector, aac, non_id_corridor_indices, corridor_names, epsilon
        )
        constraint_names += max_aac_constraint_names
        constraints += max_aac_constraints

    # NOTE: this for loop adds HVDC pair matching constraints
    if constraint_options.hvdc_matching:
        hvdc_matching_constraints, hvdc_matching_constraint_names = make_hvdc_matching_constraints(
            ntc_value_vector, hvdc_index_pairs, hvdc_ilf, corridor_names, epsilon
        )
        constraint_names += hvdc_matching_constraint_names
        constraints += hvdc_matching_constraints

    pairwise_matrix = cp.reshape(ntc_value_vector, (ntc_value_vector.shape[0] // 2, 2), "C")
    pairwise_sum = pairwise_matrix.sum(axis=1)
    for element in range(pairwise_matrix.shape[0]):
        constraints.append(pairwise_sum[element] <= 3e4)

    prob = cp.Problem(cp.Maximize(0), constraints)
    prob.solve(solver=SOLVER, **SOLVER_SETTINGS)

    verified_subset = []
    problem_subset_indices = []

    if prob.status != "optimal" and constraint_options.error_on_feasability:
        raise ValueError(" No feasible solution for problem.")
        """
        for i in range(len(ntc_atleast_aac_constraints)):
            solver_subset = verified_subset + [ntc_atleast_aac_constraints[i]]
            sub_prob = cp.Problem(cp.Maximize(0), flow_less_than_ram_constraints + solver_subset)
            sub_prob.solve(verbose=False, solver=SOLVER)# , tol_gap_abs=1e-1, tol_gap_rel=1e-1)
            if sub_prob.status != 'optimal':
                logging.getLogger().error(f'Solution failed at adding {aac_constraint_names[i]} with aac {aac[i]}')
                problem_subset_indices.append(i)
            else:
                verified_subset.append(ntc_atleast_aac_constraints[i])
        """

    return constraints, constraint_names


def make_hvdc_matching_constraints(
    ntc_value_vector: cp.Variable,
    hvdc_index_pairs: list[tuple[int, int]],
    hvdc_ilf: list[int],
    corridor_names: list[str],
    epsilon: float,
):
    hvdc_matching_constraitns = []
    hvdc_matching_constraint_names = []

    for which_hvdc, pair in enumerate(hvdc_index_pairs):
        i, j = pair
        hvdc_matching_constraitns.append(
            cp.abs(ntc_value_vector[j] - (1 - hvdc_ilf[which_hvdc]) * ntc_value_vector[i]) <= 1
        )
        hvdc_matching_constraint_names.append(f"{corridor_names[i]} HVDC matching")
    return hvdc_matching_constraitns, hvdc_matching_constraint_names


def make_ntc_max_aac_constraints(
    ntc_value_vector: cp.Variable,
    aac: NDArray,
    non_id_corridor_indices: list[int],
    corridor_names: list[str],
    epsilon: float,
):
    max_aac_constraints = []
    max_aac_constraint_names = []

    for corridor in non_id_corridor_indices:
        max_aac_constraints.append((aac[corridor] - ntc_value_vector[corridor]) >= -0.001)
        max_aac_constraint_names.append(f"{corridor_names[corridor]} at max AAC given - no ID")
    return max_aac_constraints, max_aac_constraint_names


def make_ntc_atleast_aac_constraints(
    ntc_value_vector: cp.Variable, aac: NDArray, corridor_names: list[str], epsilon: float
):
    aac_constraints: list[cp.Constraint] = []
    aac_constraint_names: list[str] = []

    for i in range(aac.shape[0]):
        # if 'SWL' in corridor_names[i]:
        #    #aac_constraints.append((ntc_value_vector[i] - aac[i]) >= -100)
        #    continue
        # else:
        aac_constraints.append((ntc_value_vector[i] - aac[i]) >= -0.001)
        # EXAMPLE                -1000              - (-1000) = 0 GOOD
        aac_constraint_names.append(f"{corridor_names[i]} at least AAC given")
    return aac_constraints, aac_constraint_names


def ntc_is_min_aac_expression(ntc_value: float | cp.Variable, aac_value: float) -> bool | cp.Constraint:
    return (ntc_value - aac_value) >= 0


def make_ram_constraints(
    ntc_value_vector: cp.Variable, ram_values: NDArray, ptdfs: NDArray, cnec_names: list[str], epsilon: float
):
    ram_constraints = []
    ram_constraint_names = []
    for i in range(
        ptdfs.shape[0]
    ):  # Iterate over each row - which represent an individual CNEC or PTC - in the ptdfs matrix
        ram_constraints.append((ram_values[i] - ptdfs[i, :] @ ntc_value_vector) >= 0)
        ram_constraint_names.append(f"{cnec_names[i]} ram remainder")
    return ram_constraints, ram_constraint_names


def optimize_ntc_values(
    constraint_options: ConstraintOptions,
    normalization: float,
    ram_values: NDArray,
    ptdfs: NDArray,
    aac: NDArray,
    hvdc_index_pairs: list[tuple[int, int]],
    hvdc_ilf: list[float],
    non_id_corridor_indices: list[int],
    corridor_names: list[str],
    cnec_names: list[str],
    allow_hail_mary: bool = False,
    allow_rerun: bool = True,
) -> tuple[cp.Problem, NDArray]:
    ntc_value_vector_length = aac.shape[0]  # Set the length of the NTC value vector based on AAC vector length
    ntc_value_vector = cp.Variable(ntc_value_vector_length, name="NTC")
    ntc_value_vector.value = np.zeros_like(aac)

    ntc_delta_vector = ntc_value_vector + aac
    # Objective function setup
    pairwise_matrix = cp.reshape(ntc_delta_vector, (ntc_value_vector_length // 2, 2), "C")
    # shape (35, 2)

    objective_filter = cp.Parameter(pairwise_matrix.shape)
    filter_array = np.ones(pairwise_matrix.shape)
    for index in non_id_corridor_indices:
        filter_array[index // 2, :] = 0

    objective_filter.value = filter_array
    pairwise_matrix = cp.multiply(pairwise_matrix, objective_filter)

    pairwise_summed_vector = cp.log(cp.sum(pairwise_matrix, axis=1) / normalization)
    objective = cp.Maximize(cp.sum(pairwise_summed_vector))

    constraints, constraint_names = make_constraints(
        constraint_options,
        ntc_value_vector,
        ram_values,
        ptdfs,
        aac,
        hvdc_index_pairs,
        hvdc_ilf,
        non_id_corridor_indices,
        corridor_names,
        cnec_names,
    )
    # check_value_feasibility(constraints, ntc_value_vector, aac, constraint_names)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=SOLVER, **SOLVER_SETTINGS)
    except cp.SolverError:
        if allow_rerun:
            constraint_options.maximum_aac_for_non_id = False
            return optimize_ntc_values(
                constraint_options,
                normalization,
                ram_values,
                ptdfs,
                aac,
                hvdc_index_pairs,
                hvdc_ilf,
                non_id_corridor_indices,
                corridor_names,
                cnec_names,
                allow_hail_mary,
                allow_rerun=False,
            )
        else:
            return prob, aac

    current_optimum = ntc_value_vector.value

    if prob.status != "optimal" and constraint_options.allow_simplified_hvdc and constraint_options.hvdc_matching:
        constraint_options.hvdc_matching = False
        constraints, constraint_names = make_constraints(
            constraint_options,
            ntc_value_vector,
            ram_values,
            ptdfs,
            aac,
            hvdc_index_pairs,
            hvdc_ilf,
            non_id_corridor_indices,
            corridor_names,
            cnec_names,
        )
        print("=========================== HVDC HANDLING =============================")
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=True, solver=SOLVER, **SOLVER_SETTINGS)
        print("=========================== HVDC END =============================")

        current_optimum = ntc_value_vector.value
        if prob.status != "optimal":
            constraint_options.hvdc_matching = True
            constraints, constraint_names = make_constraints(
                constraint_options,
                ntc_value_vector,
                ram_values,
                ptdfs,
                aac,
                hvdc_index_pairs,
                hvdc_ilf,
                non_id_corridor_indices,
                corridor_names,
                cnec_names,
            )
            prob = cp.Problem(objective, constraints)

    if prob.status != "optimal" and allow_hail_mary:
        corridor_both_directions = np.array(corridor_names).reshape(pairwise_matrix.shape)
        prob, corridors_to_exclude = hail_mary(pairwise_matrix.shape, objective_filter, prob)
        # prob = soften_hail_mary()
        if prob.status == "optimal":
            for ind in corridors_to_exclude:
                logging.getLogger().error(f"Excluding {corridor_both_directions[ind]} from OPT")
        current_optimum = ntc_value_vector.value

    if (
        prob.status != "optimal"
        and allow_hail_mary
        and constraint_options.allow_simplified_hvdc
        and constraint_options.hvdc_matching
    ):
        constraint_options.hvdc_matching = False
        constraints, constraint_names = make_constraints(
            constraint_options,
            ntc_value_vector,
            ram_values,
            ptdfs,
            aac,
            hvdc_index_pairs,
            hvdc_ilf,
            non_id_corridor_indices,
            corridor_names,
            cnec_names,
        )
        corridor_both_directions = np.array(corridor_names).reshape(pairwise_matrix.shape)
        prob, corridors_to_exclude = hail_mary(pairwise_matrix.shape, objective_filter, prob)
        # prob = soften_hail_mary()
        if prob.status == "optimal":
            for ind in corridors_to_exclude:
                logging.getLogger().error(f"Excluding {corridor_both_directions[ind]} from OPT")

        current_optimum = ntc_value_vector.value

    if not constraint_options.hvdc_matching and prob.status == "optimal" and constraint_options.allow_simplified_hvdc:
        current_optimum = harmoize_hvdc(current_optimum, hvdc_index_pairs)

    if current_optimum is None:
        optimum = aac
    else:
        optimum = current_optimum

    return prob, optimum


def harmoize_hvdc(ntc_values, hvdc_pair_indices) -> NDArray:
    # pairwise_matrix = np.reshape(ntc_values, (ntc_values.shape[0] // 2, 2))
    for index_pair in hvdc_pair_indices:
        new_value = min(abs(ntc_values[index_pair[0]]), abs(ntc_values[index_pair[1]]))
        ntc_values[index_pair[0]] = new_value
        ntc_values[index_pair[1]] = new_value
    return ntc_values


def soften_hail_mary(
    corridors_to_exclude: list[int], matrix_shape: tuple[int, ...], objective_filter: cp.Parameter, prob: cp.Problem
):
    final_corridors_to_exclude = []
    softened_corridors = []

    for i in corridors_to_exclude:
        filter_array = np.ones(matrix_shape)
        for j in corridors_to_exclude:
            if j == i:
                continue
            filter_array[i, :] = 1e-4
        objective_filter.value = filter_array
        prob.solve(solver=SOLVER, **SOLVER_SETTINGS)
        if prob.status != "optimal":
            final_corridors_to_exclude.append(i)


def hail_mary(matrix_shape: tuple[int, ...], objective_filter: cp.Parameter, prob: cp.Problem):
    corridors_to_exclude = []
    final_corridors_to_exclude = []
    filter_array = np.zeros(matrix_shape)

    for i in range(filter_array.shape[0]):
        filter_array[i, :] = 1
        objective_filter.value = filter_array
        prob.solve(solver=SOLVER, **SOLVER_SETTINGS)
        if prob.status != "optimal":
            corridors_to_exclude.append(i)
        filter_array[i, :] = 0

    if corridors_to_exclude == []:
        pairwise_faults = []
        for combination in combinations(range(filter_array.shape[0]), 2):
            filter_array[combination[0], :] = 1
            filter_array[combination[1], :] = 1
            objective_filter.value = filter_array
            prob.solve(solver=SOLVER, **SOLVER_SETTINGS)
            if prob.status != "optimal":
                pairwise_faults.append(combination)

            filter_array[combination[0], :] = 0
            filter_array[combination[1], :] = 0

    for i in corridors_to_exclude:
        filter_array = np.ones(matrix_shape)
        for j in corridors_to_exclude:
            if j == i:
                continue
            filter_array[i, :] = 0
        objective_filter.value = filter_array
        prob.solve(solver=SOLVER, **SOLVER_SETTINGS)
        if prob.status != "optimal":
            final_corridors_to_exclude.append(i)

    print("=========================== HAIL MARY =============================")

    filter_array = np.ones(matrix_shape)
    for i in final_corridors_to_exclude:
        filter_array[i, :] = 0

    objective_filter.value = filter_array
    prob.solve(verbose=True, solver=SOLVER, **SOLVER_SETTINGS)
    print("=========================== HAIL END =============================")
    return prob, final_corridors_to_exclude


def check_value_feasibility(
    constraints: list[cp.Constraint], variable: cp.Variable, values: NDArray, constraint_names: list[str]
):
    variable.value = values
    violations = 0
    for const_name, constraint in zip(constraint_names, constraints):
        if not constraint.is_dcp():
            logging.getLogger().error(const_name + " Is not DCP")
            violations += 1
        if not constraint.value():
            logging.getLogger().error(const_name + " Is not valid")
            violations += 1

    if violations > 0:
        return True
    else:
        return False


def compute_optimal_ntcs_for_mtu(
    optimization_parameters: OptimizationParameters,
    optimization_config: OptimizationConfig,
    constraint_config: ConstraintOptions,
):
    matrix_for_mtu = optimization_parameters.gc_matrix
    ram = optimization_parameters.ram
    aac = optimization_parameters.aac
    ptdfs = optimization_parameters.ptdfs
    ptdf_columns = optimization_parameters.ptdf_columns

    if matrix_for_mtu.empty:
        raise ValueError(f"No data in gc matrix for the given MTU {optimization_parameters.mtu}")

    indices_to_eliminate_from_id = []
    for code in optimization_config.codes_to_eliminate_from_id:
        indices_to_eliminate_from_id += [i for i, col in enumerate(ptdf_columns) if code in col]

    indices_to_eliminate_from_id = list(set(indices_to_eliminate_from_id))
    cnec_names = matrix_for_mtu["JAO_CNEC_Name"].loc[matrix_for_mtu["Non_Redundant"]].to_list()

    prob, solution = optimize_ntc_values(
        constraint_config,
        1,
        ram,
        ptdfs,
        aac,
        optimization_parameters.hvdc_index_pairs,
        optimization_parameters.hvdc_ilf,
        indices_to_eliminate_from_id,
        ptdf_columns,
        cnec_names,
        optimization_config.allow_hail_mary,
    )
    mtu_list = [optimization_parameters.mtu for _ in ptdf_columns]

    mtu_list_for_flows = [optimization_parameters.mtu for _ in range(ptdfs.shape[0])]
    object_flows = []
    for i in range(ptdfs.shape[0]):
        object_flows.append(np.dot(ptdfs[i, :], solution))

    return (
        NTCSolutionForMTU(
            ptdf_columns,
            mtu_list,
            solution,
            aac,
        ),
        FlowSituationforMTU(
            cnec_names,
            mtu_list_for_flows,
            ram,
            np.array(object_flows),
        ),
        prob,
    )
