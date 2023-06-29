"""'
Cost-effectiveness model of field-building programs for students
"""

"""
Imports
"""

import sys

sys.path.append("src")

# Lower-level functions called by the model
from utilities.functions.scientist_equivalent_probability import (
    ScientistEquivalentProbability,
)
import utilities.functions.n_participant as npart
import utilities.sampling.model_utilities as mu
import utilities.functions.qarys_over_t as qot

# Vectorization
import numpy as np

# Gamma distributions
import squigglepy as sq


"""
Cost function
"""


def cost(params, params_sampled, n_sim):
    """
    Calculate the cost of supporting a student program.

    Args:
        params (dict): Dictionary of calibrated parameters.
        params_sampled (dict): Dictionary of sampled parameters.

    Returns:
        tuple: Updated dictionaries for params and params_sampled, and a derived_params_sampled dictionary.
    """

    # Calculate fixed cost of labor
    fixed_cost_labor = (
        params_sampled["fixed_hours_labor"] * params_sampled["average_wage"]
    )

    # Calculate total fixed cost
    fixed_cost = fixed_cost_labor + params_sampled["fixed_cost_other"]

    # Calculate target variable cost
    target_variable_cost = np.where(
        (fixed_cost < params_sampled["target_budget"])
        & (np.mean(fixed_cost) < params_sampled["target_budget"]),
        params_sampled["target_budget"] - np.mean(fixed_cost),
        0,
    )

    # Calculate actual variable cost spent on students
    target_variable_cost_students = (
        target_variable_cost * params_sampled["split_variable_cost_students"]
    )
    mask = target_variable_cost_students == 0
    target_variable_cost_students_no_zero = np.where(
        mask, 1, target_variable_cost_students
    )

    k_variable_cost_students = (
        target_variable_cost_students_no_zero / params["sd_variable_cost_students"]
    ) ** 2
    theta_variable_cost_students = (
        params["sd_variable_cost_students"] ** 2
    ) / target_variable_cost_students_no_zero
    actual_variable_cost_students_unmasked = (
        sq.gamma(shape=k_variable_cost_students, scale=theta_variable_cost_students)
        @ n_sim
    )
    actual_variable_cost_students = np.where(
        mask, 0, actual_variable_cost_students_unmasked
    )

    # Calculate actual variable cost spent on labor
    target_variable_cost_labor = target_variable_cost - target_variable_cost_students
    mask = target_variable_cost == 0
    target_variable_cost_labor_no_zero = np.where(mask, 1, target_variable_cost_labor)

    target_hours_labor = (
        target_variable_cost_labor_no_zero / params_sampled["average_wage"]
    )

    k_hours_labor = (target_hours_labor / params["sd_hours_labor"]) ** 2
    theta_hours_labor = (params["sd_hours_labor"] ** 2) / target_hours_labor
    actual_labor_hours_unmasked = (
        sq.gamma(shape=k_hours_labor, scale=theta_hours_labor) @ n_sim
    )
    actual_labor_hours = np.where(mask, 0, actual_labor_hours_unmasked)

    actual_variable_cost_labor = actual_labor_hours * params_sampled["average_wage"]

    # Aggregate costs back up
    actual_variable_cost = actual_variable_cost_students + actual_variable_cost_labor

    actual_budget = fixed_cost + actual_variable_cost

    # Update derived_params_sampled dictionary
    derived_params_sampled = {
        "fixed_cost_labor": fixed_cost_labor,
        "fixed_cost": fixed_cost,
        "target_variable_cost": target_variable_cost,
        "target_variable_cost_students": target_variable_cost_students,
        "target_variable_cost_labor": target_variable_cost_labor,
        "actual_labor_hours": actual_labor_hours,
        "actual_variable_cost_students": actual_variable_cost_students,
        "actual_variable_cost_labor": actual_variable_cost_labor,
        "actual_variable_cost": actual_variable_cost,
        "actual_budget": actual_budget,
    }

    # Return updated dictionaries
    return params, params_sampled, derived_params_sampled


"""
Benefit functions
"""


def benefit_number_participant(params, params_sampled, derived_params_sampled):
    """
    Calculates the number of people-related benefits of a student program.
    This includes the expected number of scientist-equivalents.

    Args:
        params (dict): Input parameters for the calculation.
        params_sampled (dict): Sampled parameters for the calculation.
        derived_params_sampled (dict): Derived sampled parameters for the calculation.

    Returns:
        tuple: A tuple containing the following elements:
            - params (dict): Original input parameters.
            - params_sampled (dict): Sampled parameters for the calculation.
            - derived_params_sampled (dict): Derived sampled parameters for the calculation, including additional calculations.
    """

    # Initialize dictionary
    derived_params_sampled_phd = {}

    # Calculate number of undergraduates
    n_student_undergrad = npart.compute_n_student(
        derived_params_sampled["actual_variable_cost"],
        params["n_student_undergrad_scaling_parameter_gamma"],
        params["n_student_undergrad_scaling_parameter_slope"],
        params["n_student_undergrad_scaling_parameter_intercept"],
        params["n_student_deterministic"],
    )

    # Calculate pipeline probabilities
    sep = ScientistEquivalentProbability(
        params_sampled["p_scientist_given_phd"],
        params_sampled["p_professor_given_phd"],
        params_sampled["p_engineer_given_phd"],
        params_sampled["scientist_equivalent_professor"],
        params_sampled["scientist_equivalent_engineer"],
        params_sampled["p_phd_given_pursue_ais"],
        params_sampled["p_pursue_ais"],
        params_sampled["p_scientist_given_not_phd"],
        params_sampled["p_professor_given_not_phd"],
        params_sampled["p_engineer_given_not_phd"],
    )

    # Calculate scientist-equivalents for undergrads with vs. without PhD
    n_scientist_equivalent_student_undergrad_via_phd_during = (
        n_student_undergrad * sep.p_phd * params_sampled["scientist_equivalent_phd"]
    )
    n_scientist_equivalent_student_undergrad_via_phd_after = (
        n_student_undergrad * sep.p_scientist_equivalent_via_phd
    )
    n_scientist_equivalent_student_undergrad_not_via_phd = (
        n_student_undergrad * sep.p_scientist_equivalent_not_via_phd
    )

    # Calculate total scientist-equivalents for undergrads
    n_scientist_equivalent_student_undergrad_during = (
        n_scientist_equivalent_student_undergrad_via_phd_during
        + n_scientist_equivalent_student_undergrad_not_via_phd
    )
    n_scientist_equivalent_student_undergrad_after = (
        n_scientist_equivalent_student_undergrad_via_phd_after
        + n_scientist_equivalent_student_undergrad_not_via_phd
    )

    # If program includes PhD students, calculate them too
    if params["student_phd"] == True:
        n_student_phd = npart.compute_n_student(
            derived_params_sampled["actual_variable_cost"],
            params["n_student_phd_scaling_parameter_gamma"],
            params["n_student_phd_scaling_parameter_slope"],
            params["n_student_phd_scaling_parameter_intercept"],
            params["n_student_deterministic"],
        )

        # Calculate scientist-equivalents for PhDs
        n_scientist_equivalent_student_phd_during = (
            n_student_phd * params_sampled["scientist_equivalent_phd"]
        )
        n_scientist_equivalent_student_phd_after = (
            n_student_phd * sep.p_scientist_equivalent_given_phd
        )

        # Update derived_params_sampled dictionary
        derived_params_sampled_phd = {
            "n_student_phd": n_student_phd,
            "n_scientist_equivalent_student_phd_during": n_scientist_equivalent_student_phd_during,
            "n_scientist_equivalent_student_phd_after": n_scientist_equivalent_student_phd_after,
        }

    # output
    derived_params_sampled = {
        **derived_params_sampled,
        **derived_params_sampled_phd,
        "p_scientist_equivalent_given_phd": sep.p_scientist_equivalent_given_phd,
        "p_scientist_equivalent_given_not_phd": sep.p_scientist_equivalent_given_not_phd,
        "p_scientist_equivalent_via_phd": sep.p_scientist_equivalent_via_phd,
        "p_scientist_equivalent_not_via_phd": sep.p_scientist_equivalent_not_via_phd,
        "p_scientist_equivalent": sep.p_scientist_equivalent,
        "n_student_undergrad": n_student_undergrad,
        "n_scientist_equivalent_student_undergrad_via_phd_during": n_scientist_equivalent_student_undergrad_via_phd_during,
        "n_scientist_equivalent_student_undergrad_via_phd_after": n_scientist_equivalent_student_undergrad_via_phd_after,
        "n_scientist_equivalent_student_undergrad_not_via_phd": n_scientist_equivalent_student_undergrad_not_via_phd,
        "n_scientist_equivalent_student_undergrad_during": n_scientist_equivalent_student_undergrad_during,
        "n_scientist_equivalent_student_undergrad_after": n_scientist_equivalent_student_undergrad_after,
    }

    # Return updated dictionaries
    return params, params_sampled, derived_params_sampled


def benefit_qarys_per_participant(params, params_sampled, derived_params_sampled):
    """
    Calculates the quality-adjusted research hours (QARYs) per participant and
    the total QARYs for the calculation.

    Args:
        params (dict), params_sampled (dict), derived_params_sampled (dict)

    Returns:
        tuple: A tuple containing the following elements:
            - params (dict): Original input parameters.
            - params_sampled (dict): Sampled parameters for the calculation.
            - derived_params_sampled (dict): Derived sampled parameters for the calculation, including additional calculations.
            - derived_functions (dict): Derived functions for the calculation.
    """

    # Initialize output dictionaries
    derived_functions = {}

    # Dictionaries to combine
    derived_params_sampled_student_list = []
    derived_functions_student_list = []

    # Determine student types
    student_types = ["_undergrad_via_phd", "_undergrad_not_via_phd"]
    if params["student_phd"] == True:
        student_types.append("_phd")

    # Calculate QARYs for each student type
    for student_type in student_types:
        (
            params,
            params_sampled,
            derived_params_sampled_student,
            derived_functions_student,
        ) = qot.benefit_qarys_over_time(
            params,
            params_sampled,
            derived_params_sampled,
            participant_type="_student",
            researcher_type=student_type,
        )

        derived_params_sampled_student_list.append(derived_params_sampled_student)
        derived_functions_student_list.append(derived_functions_student)

    # Combine dictionaries across researcher types
    (
        derived_params_sampled_student,
        derived_functions,
    ) = qot.combine_dicts_across_researcher_types(
        derived_params_sampled_student_list,
        derived_functions_student_list,
        participant="student",
        researcher_types=[s[1:] for s in student_types],
    )

    # Combine dictionaries
    derived_params_sampled = {
        **derived_params_sampled,
        **derived_params_sampled_student,
    }

    # Total QARYs
    derived_params_sampled["qarys"] = derived_params_sampled["qarys_student"]

    # Output
    return params, params_sampled, derived_params_sampled, derived_functions


def benefit_mfn(params, params_sampled, derived_params_sampled):
    """
    This function calculates benefit and its component parts.

    Args:
        params (dict): Input parameters for the calculation.
        params_sampled (dict): Sampled parameters for the calculation.
        derived_params_sampled (dict): Derived sampled parameters for the calculation.

    Returns:
        tuple: A tuple containing the following elements:
            - params (dict): Original input parameters.
            - params_sampled (dict): Sampled parameters for the calculation.
            - derived_params_sampled (dict): Derived sampled parameters for the calculation, including additional calculations.
    """

    # Calculate the number of participants
    params, params_sampled, derived_params_sampled = benefit_number_participant(
        params, params_sampled, derived_params_sampled
    )

    # Calculate QARYs per participant
    (
        params,
        params_sampled,
        derived_params_sampled,
        derived_functions,
    ) = benefit_qarys_per_participant(params, params_sampled, derived_params_sampled)

    # Return the updated parameters, sampled parameters, new sampled parameters, and new functions
    return params, params_sampled, derived_params_sampled, derived_functions


"""
Master function
"""


def mfn(params, params_cf, n_sim):
    """
    This is the final function to which parameters are fed. It calculates and
    compares the benefits under the program and the counterfactual, and it also
    calculates the simulated outputs.

    Args:
        params (dict): Input parameters for the calculation.
        params_cf (dict): Counterfactual input parameters for the calculation.
        n_sim (int): Number of simulations.

    Returns:
        tuple: A tuple containing the following elements:
            - params (dict): Original input parameters.
            - params_sampled (dict): Sampled parameters for the calculation.
            - derived_params_sampled (dict): Derived sampled parameters for the calculation, including additional calculations.
            - derived_functions (dict): Derived functions for the calculation.
    """

    # Sample given parameters
    params, params_sampled = mu.sample_params(params, n_sim)
    params_cf, params_sampled_cf = mu.sample_params(params_cf, n_sim)

    # Calculate cost
    params, params_sampled, derived_params_sampled = cost(params, params_sampled, n_sim)
    derived_params_sampled_cf = derived_params_sampled

    # Calculate benefit
    params, params_sampled, derived_params_sampled, derived_functions = benefit_mfn(
        params, params_sampled, derived_params_sampled
    )
    (
        params_cf,
        params_sampled_cf,
        derived_params_sampled_cf,
        derived_functions_cf,
    ) = benefit_mfn(params_cf, params_sampled_cf, derived_params_sampled_cf)

    # Combine new parameters with the previous parameters
    params_sampled = mu.merge_params_with_cf(params_sampled, params_sampled_cf)
    derived_params_sampled = mu.merge_params_with_cf(
        derived_params_sampled, derived_params_sampled_cf
    )
    derived_functions = mu.merge_params_with_cf(derived_functions, derived_functions_cf)

    # Return both the original and the new parameters
    return params, params_sampled, derived_params_sampled, derived_functions
