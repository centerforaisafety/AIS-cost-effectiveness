"""
Cost-effectiveness model of field-building programs for research professionals
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
    Calculate the cost of a professional program.

    The function calculates the fixed cost of labor,total fixed cost,
    variable cost, and checks if the fixed cost exceeds the budget.

    Args:
        params (dict): Input parameters for the calculation.
        params_sampled (dict): Sampled parameters for the calculation.

    Returns:
        tuple: A tuple containing the following elements:
            - params (dict): Original input parameters.
            - params_sampled (dict): Sampled parameters for the calculation.
            - derived_params_sampled (dict): Derived sampled parameters for the calculation, including additional calculations.
    """

    # Calculate the fixed costs of the labor
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

    # Initialize output values
    actual_variable_cost_event = np.zeros(n_sim)
    actual_variable_cost_award = np.zeros(n_sim)

    derived_params_sampled_attendee = {}
    derived_params_sampled_contender = {}

    # Split target budget spent on event cost
    if params["participant_attendee"]:
        target_variable_cost_event = (
            target_variable_cost * params_sampled["split_variable_cost_event"]
        )

        # Create a mask where target_variable_cost_event equals zero
        mask = target_variable_cost_event == 0
        target_variable_cost_event_no_zero = np.where(
            mask, 1, target_variable_cost_event
        )

        k_variable_cost_event = (
            target_variable_cost_event_no_zero / params["sd_variable_cost_event"]
        ) ** 2
        theta_variable_cost_event = (
            params["sd_variable_cost_event"] ** 2
        ) / target_variable_cost_event_no_zero
        actual_variable_cost_event_unmasked = (
            sq.gamma(shape=k_variable_cost_event, scale=theta_variable_cost_event)
            @ n_sim
        )
        actual_variable_cost_event = np.where(
            mask, 0, actual_variable_cost_event_unmasked
        )

        derived_params_sampled_attendee = {
            "target_variable_cost_event": target_variable_cost_event,
            "actual_variable_cost_event": actual_variable_cost_event,
        }

    # Split target budget spent on award cost
    if params["participant_contender"]:
        target_variable_cost_award = (
            target_variable_cost * params_sampled["split_variable_cost_award"]
        )

        # Create a mask where target_variable_cost_award equals zero
        mask = target_variable_cost_award == 0
        target_variable_cost_award_no_zero = np.where(
            mask, 1, target_variable_cost_award
        )

        k_variable_cost_award = (
            target_variable_cost_award_no_zero / params["sd_variable_cost_award"]
        ) ** 2
        theta_variable_cost_award = (
            params["sd_variable_cost_award"] ** 2
        ) / target_variable_cost_award
        actual_variable_cost_award_unmasked = (
            sq.gamma(shape=k_variable_cost_award, scale=theta_variable_cost_award)
            @ n_sim
        )
        actual_variable_cost_award = np.where(
            mask, 0, actual_variable_cost_award_unmasked
        )

        derived_params_sampled_contender = {
            "target_variable_cost_award": target_variable_cost_award,
            "actual_variable_cost_award": actual_variable_cost_award,
        }

    # Split target budget spent on labor
    split_variable_cost_labor = (
        1
        - params_sampled["split_variable_cost_event"]
        - params_sampled["split_variable_cost_award"]
    )
    target_variable_cost_labor = target_variable_cost * split_variable_cost_labor

    # Create a mask where target_variable_cost_award equals zero
    mask = target_variable_cost_labor == 0
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

    # Aggregate costs back up
    actual_variable_cost_labor = actual_labor_hours * params_sampled["average_wage"]
    actual_variable_cost = (
        actual_variable_cost_event
        + actual_variable_cost_award
        + actual_variable_cost_labor
    )

    actual_budget = fixed_cost + actual_variable_cost

    # Output
    derived_params_sampled = {
        **derived_params_sampled_attendee,
        **derived_params_sampled_contender,
        "fixed_cost_labor": fixed_cost_labor,
        "fixed_cost": fixed_cost,
        "target_variable_cost": target_variable_cost,
        "target_variable_cost_labor": target_variable_cost_labor,
        "actual_labor_hours": actual_labor_hours,
        "actual_variable_cost_labor": actual_variable_cost_labor,
        "actual_variable_cost": actual_variable_cost,
        "actual_budget": actual_budget,
    }

    return params, params_sampled, derived_params_sampled


"""
Benefit functions
"""


def benefit_number_participant(params, params_sampled, derived_params_sampled):
    """
    Calculates the number of people-related benefits of a professional program.

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

    # Initialize the output variables
    n_entry = 0

    n_attendee = 0
    n_scientist_equivalent_attendee = 0

    n_contender = 0
    n_scientist_equivalent_contender = 0

    derived_params_sampled_attendee = {}
    derived_params_sampled_attendee_event_capacity = {}

    derived_params_sampled_contender = {}

    # Calculate pipeline probabilities
    sep = ScientistEquivalentProbability(
        params_sampled["p_scientist_given_phd"],
        params_sampled["p_professor_given_phd"],
        params_sampled["p_engineer_given_phd"],
        params_sampled["scientist_equivalent_professor"],
        params_sampled["scientist_equivalent_engineer"],
        0,
        0,
        0,
        0,
    )

    if params["participant_attendee"] == True:
        # Max capacity for truncation
        derived_params_sampled["actual_variable_cost_event"] = np.where(
            derived_params_sampled["actual_variable_cost_event"] <= 1,
            1,
            derived_params_sampled["actual_variable_cost_event"],
        )
        max_capacity = (
            derived_params_sampled["actual_variable_cost_event"] / 1000
        ) * params_sampled["event_max_capacity_per_1000"]

        # Number of attendees in total
        n_attendee = npart.compute_n_attendee(
            derived_params_sampled["actual_variable_cost_event"],
            max_capacity,
            gamma=params["n_attendee_scaling_parameter_gamma"],
            slope=params["n_attendee_scaling_parameter_slope"],
            intercept=params["n_attendee_scaling_parameter_intercept"],
        )

        # Number of attendees of each researcher type
        n_attendee_scientist = (
            n_attendee * params_sampled["fraction_attendee_scientist"]
        )
        n_attendee_professor = (
            n_attendee * params_sampled["fraction_attendee_professor"]
        )
        n_attendee_engineer = n_attendee * params_sampled["fraction_attendee_engineer"]
        fraction_attendee_phd = 1 - (
            params_sampled["fraction_attendee_scientist"]
            + params_sampled["fraction_attendee_professor"]
            + params_sampled["fraction_attendee_engineer"]
        )
        n_attendee_phd = n_attendee * fraction_attendee_phd

        # Converting to scientist-equivalents - attendees
        n_scientist_equivalent_attendee_scientist = n_attendee_scientist
        n_scientist_equivalent_attendee_professor = (
            n_attendee_professor * params_sampled["scientist_equivalent_professor"]
        )

        n_scientist_equivalent_attendee_engineer = (
            n_attendee_engineer * params_sampled["scientist_equivalent_engineer"]
        )

        n_scientist_equivalent_attendee_phd_during = (
            n_attendee_phd * params_sampled["scientist_equivalent_phd"]
        )

        n_scientist_equivalent_attendee_phd_after = (
            n_attendee_phd * sep.p_scientist_equivalent_given_phd
        )

        n_scientist_equivalent_attendee = (
            n_scientist_equivalent_attendee_scientist
            + n_scientist_equivalent_attendee_professor
            + n_scientist_equivalent_attendee_engineer
            + n_scientist_equivalent_attendee_phd_during
        )

        # Output
        derived_params_sampled_attendee = {
            **derived_params_sampled_attendee_event_capacity,
            "max_capacity": max_capacity,
            "n_attendee": n_attendee,
            "n_attendee_scientist": n_attendee_scientist,
            "n_attendee_professor": n_attendee_professor,
            "n_attendee_engineer": n_attendee_engineer,
            "fraction_attendee_phd": fraction_attendee_phd,
            "n_attendee_phd": n_attendee_phd,
            "n_attendee_phd_during": n_attendee_phd,
            "n_attendee_phd_after": n_attendee_phd,
            "n_scientist_equivalent_attendee_scientist": n_scientist_equivalent_attendee_scientist,
            "n_scientist_equivalent_attendee_professor": n_scientist_equivalent_attendee_professor,
            "n_scientist_equivalent_attendee_engineer": n_scientist_equivalent_attendee_engineer,
            "n_scientist_equivalent_attendee_phd_during": n_scientist_equivalent_attendee_phd_during,
            "n_scientist_equivalent_attendee_phd_after": n_scientist_equivalent_attendee_phd_after,
            "n_scientist_equivalent_attendee": n_scientist_equivalent_attendee,
        }

    if params["participant_contender"] == True:
        # Calculate number of entries
        n_entry = npart.compute_n_entry(
            derived_params_sampled["actual_variable_cost_award"],
            gamma=params["n_contender_scaling_parameter_gamma"],
            slope=params["n_contender_scaling_parameter_slope"],
            intercept=params["n_contender_scaling_parameter_intercept"],
        )

        # Number of contenders in total
        n_contender_scientist = n_entry * params_sampled["n_scientist_per_entry"]
        n_contender_professor = n_entry * params_sampled["n_professor_per_entry"]
        n_contender_engineer = n_entry * params_sampled["n_engineer_per_entry"]
        n_contender_phd = n_entry * params_sampled["n_phd_per_entry"]

        n_contender = (
            n_contender_scientist
            + n_contender_professor
            + n_contender_engineer
            + n_contender_phd
        )

        # Converting to scientist-equivalents - contenders
        n_scientist_equivalent_contender_scientist = n_contender_scientist
        n_scientist_equivalent_contender_professor = (
            n_contender_professor * params_sampled["scientist_equivalent_professor"]
        )
        n_scientist_equivalent_contender_engineer = (
            n_contender_engineer * params_sampled["scientist_equivalent_engineer"]
        )
        n_scientist_equivalent_contender_phd_during = (
            n_contender_phd * params_sampled["scientist_equivalent_phd"]
        )
        n_scientist_equivalent_contender_phd_after = (
            n_contender_phd * sep.p_scientist_equivalent_given_phd
        )
        n_scientist_equivalent_contender = (
            n_contender_scientist
            + n_scientist_equivalent_contender_professor
            + n_scientist_equivalent_contender_engineer
            + n_scientist_equivalent_contender_phd_during
        )

        # Output
        derived_params_sampled_contender = {
            "n_entry": n_entry,
            "n_contender_scientist": n_contender_scientist,
            "n_contender_professor": n_contender_professor,
            "n_contender_engineer": n_contender_engineer,
            "n_contender_phd": n_contender_phd,
            "n_contender_phd_during": n_contender_phd,
            "n_contender_phd_after": n_contender_phd,
            "n_contender": n_contender,
            "n_scientist_equivalent_contender_scientist": n_scientist_equivalent_contender_scientist,
            "n_scientist_equivalent_contender_professor": n_scientist_equivalent_contender_professor,
            "n_scientist_equivalent_contender_engineer": n_scientist_equivalent_contender_engineer,
            "n_scientist_equivalent_contender_phd_during": n_scientist_equivalent_contender_phd_during,
            "n_scientist_equivalent_contender_phd_after": n_scientist_equivalent_contender_phd_after,
            "n_scientist_equivalent_contender": n_scientist_equivalent_contender,
        }

    if (
        params["participant_attendee"] == True
        and params["participant_contender"] == True
    ):
        n_attendee = n_attendee * (1 - params["fraction_attendee_also_contender"])
        n_attendee_scientist = n_attendee_scientist * (
            1 - params["fraction_attendee_also_contender"]
        )
        n_attendee_professor = n_attendee_professor * (
            1 - params["fraction_attendee_also_contender"]
        )
        n_attendee_engineer = n_attendee_engineer * (
            1 - params["fraction_attendee_also_contender"]
        )
        n_attendee_phd = n_attendee_phd * (
            1 - params["fraction_attendee_also_contender"]
        )

        n_scientist_equivalent_attendee = n_scientist_equivalent_attendee * (
            1 - params["fraction_attendee_also_contender"]
        )
        n_scientist_equivalent_attendee_scientist = (
            n_scientist_equivalent_attendee_scientist
            * (1 - params["fraction_attendee_also_contender"])
        )
        n_scientist_equivalent_attendee_professor = (
            n_scientist_equivalent_attendee_professor
            * (1 - params["fraction_attendee_also_contender"])
        )
        n_scientist_equivalent_attendee_engineer = (
            n_scientist_equivalent_attendee_engineer
            * (1 - params["fraction_attendee_also_contender"])
        )
        n_scientist_equivalent_attendee_phd_during = (
            n_scientist_equivalent_attendee_phd_during
            * (1 - params["fraction_attendee_also_contender"])
        )
        n_scientist_equivalent_attendee_phd_after = (
            n_scientist_equivalent_attendee_phd_after
            * (1 - params["fraction_attendee_also_contender"])
        )

        # Update the output dictionary
        derived_params_sampled_attendee = {
            **derived_params_sampled_attendee,
            "n_attendee": n_attendee,
            "n_attendee_scientist": n_attendee_scientist,
            "n_attendee_professor": n_attendee_professor,
            "n_attendee_engineer": n_attendee_engineer,
            "n_attendee_phd": n_attendee_phd,
            "n_attendee_phd_during": n_attendee_phd,
            "n_attendee_phd_after": n_attendee_phd,
            "n_scientist_equivalent_attendee_scientist": n_scientist_equivalent_attendee_scientist,
            "n_scientist_equivalent_attendee_professor": n_scientist_equivalent_attendee_professor,
            "n_scientist_equivalent_attendee_engineer": n_scientist_equivalent_attendee_engineer,
            "n_scientist_equivalent_attendee_phd_during": n_scientist_equivalent_attendee_phd_during,
            "n_scientist_equivalent_attendee_phd_after": n_scientist_equivalent_attendee_phd_after,
            "n_scientist_equivalent_attendee": n_scientist_equivalent_attendee,
        }

    # Merge output dictionaries
    derived_params_sampled = {
        **derived_params_sampled,
        "p_scientist_equivalent_given_phd": sep.p_scientist_equivalent_given_phd,
        **derived_params_sampled_attendee,
        **derived_params_sampled_contender,
    }

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
    derived_params_sampled_attendee = {}
    derived_params_sampled_contender = {}

    derived_functions_attendee = {}
    derived_functions_contender = {}

    if params["participant_attendee"] == True:
        # Dictionaries to combine
        derived_params_sampled_attendee_list = []
        derived_functions_attendee_list = []

        for researcher_type in ["_scientist", "_professor", "_engineer", "_phd"]:
            (
                params,
                params_sampled,
                derived_params_sampled_attendee_researcher,
                derived_functions_attendee_researcher,
            ) = qot.benefit_qarys_over_time(
                params,
                params_sampled,
                derived_params_sampled,
                participant_type="_attendee",
                researcher_type=researcher_type,
            )

            derived_params_sampled_attendee_list.append(
                derived_params_sampled_attendee_researcher
            )
            derived_functions_attendee_list.append(
                derived_functions_attendee_researcher
            )

        # Combine dictionaries across researcher types
        (
            derived_params_sampled_attendee,
            derived_functions_attendee,
        ) = qot.combine_dicts_across_researcher_types(
            derived_params_sampled_attendee_list,
            derived_functions_attendee_list,
            participant="attendee",
        )

    if params["participant_contender"] == True:
        # Dictionaries to combine
        derived_params_sampled_contender_list = []
        derived_functions_contender_list = []

        for researcher_type in ["_scientist", "_professor", "_engineer", "_phd"]:
            (
                params,
                params_sampled,
                derived_params_sampled_contender_researcher,
                derived_functions_contender_researcher,
            ) = qot.benefit_qarys_over_time(
                params,
                params_sampled,
                derived_params_sampled,
                participant_type="_contender",
                researcher_type=researcher_type,
            )

            derived_params_sampled_contender_list.append(
                derived_params_sampled_contender_researcher
            )
            derived_functions_contender_list.append(
                derived_functions_contender_researcher
            )

        # Combine dictionaries across researcher types
        (
            derived_params_sampled_contender,
            derived_functions_contender,
        ) = qot.combine_dicts_across_researcher_types(
            derived_params_sampled_contender_list, derived_functions_contender_list
        )

    # Combine researcher type dictionaries across participants
    derived_params_sampled = {
        **derived_params_sampled,
        **derived_params_sampled_attendee,
        **derived_params_sampled_contender,
    }
    derived_functions = {**derived_functions_attendee, **derived_functions_contender}

    # Total QARYs
    derived_params_sampled["qarys"] = derived_params_sampled.get(
        "qarys_attendee", 0
    ) + derived_params_sampled.get("qarys_contender", 0)

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
    # Run previous benefit functions
    (params, params_sampled, derived_params_sampled) = benefit_number_participant(
        params, params_sampled, derived_params_sampled
    )
    (
        params,
        params_sampled,
        derived_params_sampled,
        derived_functions,
    ) = benefit_qarys_per_participant(params, params_sampled, derived_params_sampled)

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
