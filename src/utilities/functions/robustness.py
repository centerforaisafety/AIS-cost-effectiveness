"""
Purpose: functions for changing parameter specifications for robustness analysis
"""


"""
Imports
"""

import utilities.defaults.robustness as ar
import squigglepy as sq

"""
Helper functions
"""


def manipulate_mean(params, key, shift):
    """
    Manipulate the mean of the beta distribution associated with a key in params.

    Args:
        params : dict
            The params dictionary.
        key : str
            The key in the params dictionary that contains the beta distribution.
        shift : float
            The shift factor to apply to the mean.

    Returns:
        updated_beta : sq.beta
            The updated beta distribution.
    """
    a = params[key].a
    b = params[key].b
    mean = a / (a + b)
    new_mean = min(max(mean * shift, 0), 1)

    # Maintain the same value of (a+b) while updating the mean
    new_a = new_mean * (a + b)
    new_b = (1 - new_mean) * (a + b)
    updated_beta = sq.beta(new_a, new_b)

    return updated_beta


"""
Functions
"""


def update_scientist_equivalence(params, params_cf, option):
    """
    Update the scientist equivalence parameters for the given params and params_cf dictionaries.

    Args:
        params : dict
            The main params dictionary.
        params_cf : dict
            The counterfactual params dictionary.
        option : str
            The option for updating scientist equivalence, either 'smaller' or 'larger'.

    Returns:
        updated_params : dict
            The updated main params dictionary.
        updated_params_cf : dict
            The updated counterfactual params dictionary.

    Warning messages:
        ValueError
            If the option provided is not 'smaller' or 'larger'.
    """
    if option == "smaller":
        shift = 1 / ar.mid_shift
    elif option == "larger":
        shift = ar.mid_shift
    else:
        raise ValueError("Invalid option. Choose either 'smaller' or 'larger'.")

    updated_params = params.copy()
    updated_params["scientist_equivalent_professor"] *= shift
    updated_params["scientist_equivalent_engineer"] /= shift
    updated_params["scientist_equivalent_phd"] /= shift

    updated_params_cf = params_cf.copy()
    updated_params_cf["scientist_equivalent_professor"] *= shift
    updated_params_cf["scientist_equivalent_engineer"] /= shift
    updated_params_cf["scientist_equivalent_phd"] /= shift

    return updated_params, updated_params_cf


def update_job_prospects(params, params_cf, option):
    """
    Update the probability of becoming a researcher type parameters for the given params and params_cf dictionaries.

    Args:
        params : dict
            The main params dictionary.
        params_cf : dict
            The counterfactual params dictionary.
        option : str
            The option for updating probability of becoming a researcher type, either 'worse' or 'better'.

    Returns:
        updated_params : dict
            The updated main params dictionary.
        updated_params_cf : dict
            The updated counterfactual params dictionary.

    Warning messages:
        ValueError
            If the option provided is not 'worse' or 'better'.
    """
    if option == "worse":
        shift = 1 / ar.tiny_shift
    elif option == "better":
        shift = ar.tiny_shift
    else:
        raise ValueError("Invalid option. Choose either 'smaller' or 'larger'.")

    updated_params = params.copy()
    updated_params["p_scientist_given_phd"] *= shift
    updated_params["p_professor_given_phd"] *= shift
    updated_params["p_engineer_given_phd"] *= shift
    updated_params["p_scientist_given_not_phd"] *= shift
    updated_params["p_professor_given_not_phd"] *= shift
    updated_params["p_engineer_given_not_phd"] *= shift

    updated_params_cf = params_cf.copy()
    updated_params_cf["p_scientist_given_phd"] *= shift
    updated_params_cf["p_professor_given_phd"] *= shift
    updated_params_cf["p_engineer_given_phd"] *= shift
    updated_params_cf["p_scientist_given_not_phd"] *= shift
    updated_params_cf["p_professor_given_not_phd"] *= shift
    updated_params_cf["p_engineer_given_not_phd"] *= shift

    return updated_params, updated_params_cf


def update_talent_spotting(params, params_cf, option, type="_student_undergrad"):
    """
    Update ability_pivot_point parameters for the given params and params_cf dictionaries.

    Args:
        params : dict
            The main params dictionary.
        params_cf : dict
            The counterfactual params dictionary.
        option : str
            The option for updating ability_pivot_point, either 'worse' or 'better'.

    Returns:
        updated_params : dict
            The updated main params dictionary.
        updated_params_cf : dict
            The updated counterfactual params dictionary.

    Warning messages:
        ValueError
            If the option provided is not 'worse' or 'better'.
    """
    if option == "worse":
        shift = 1 / ar.mid_shift
    elif option == "better":
        shift = ar.mid_shift
    else:
        raise ValueError("Invalid option. Choose either 'smaller' or 'larger'.")

    updated_params = params.copy()
    updated_params["ability_pivot_point" + type] *= shift

    updated_params_cf = params_cf.copy()
    updated_params_cf["ability_pivot_point" + type] /= shift

    return updated_params, updated_params_cf


def update_labor(params, params_cf, option):
    """
    Update the labor parameters for the given params and params_cf dictionaries.

    Args:
        params : dict
            The main params dictionary.
        params_cf : dict
            The counterfactual params dictionary.
        option : str
            The option for updating labor, either 'more' or 'less'.

    Returns:
        updated_params : dict
            The updated main params dictionary.
        updated_params_cf : dict
            The updated counterfactual params dictionary.

    Warning messages:
        ValueError
            If the option provided is not 'more' or 'less'.
    """
    if option == "larger":
        hours_shift = ar.mid_shift
        wage_shift = ar.tiny_shift
    elif option == "smaller":
        hours_shift = 1 / ar.mid_shift
        wage_shift = 1 / ar.tiny_shift
    else:
        raise ValueError("Invalid option. Choose either 'more' or 'less'.")

    updated_params = params.copy()
    updated_params["fixed_hours_labor"] *= hours_shift
    updated_params["average_wage"] *= wage_shift

    updated_params_cf = params_cf.copy()
    updated_params_cf["fixed_hours_labor"] *= hours_shift
    updated_params_cf["average_wage"] *= wage_shift

    return updated_params, updated_params_cf


def update_fixed_cost_other(params, params_cf, option):
    """
    Update the fixed_cost_other parameter for the given params and params_cf dictionaries.

    Args:
        params : dict
            The main params dictionary.
        params_cf : dict
            The counterfactual params dictionary.
        option : str
            The option for updating fixed_cost_other, either 'larger' or 'smaller'.

    Returns
        updated_params : dict
            The updated main params dictionary.
        updated_params_cf : dict
            The updated counterfactual params dictionary.

    Warning messages:
        ValueError
            If the option provided is not 'larger' or 'smaller'.
    """
    if option == "larger":
        cost_shift = ar.mid_shift
    elif option == "smaller":
        cost_shift = 1 / ar.mid_shift
    else:
        raise ValueError("Invalid option. Choose either 'larger' or 'smaller'.")

    updated_params = params.copy()
    updated_params["fixed_cost_other"] *= cost_shift

    updated_params_cf = params_cf.copy()
    updated_params_cf["fixed_cost_other"] *= cost_shift

    return updated_params, updated_params_cf


def update_number_per_entry(params, params_cf, option):
    """
    Update the number of scientists, professors, engineers, and PhDs per entry for the given params and params_cf dictionaries.

    Args:
        params : dict
            The main params dictionary.
        params_cf : dict
            The counterfactual params dictionary.
        option : str
            The option for updating the number per entry, either 'more' or 'fewer'.

    Returns
        updated_params : dict
            The updated main params dictionary.
        updated_params_cf : dict
            The updated counterfactual params dictionary.

    Warning messages:
        ValueError
            If the option provided is not 'more' or 'less'.
    """
    if option == "more":
        shift = ar.small_shift
    elif option == "fewer":
        shift = 1 / ar.small_shift
    else:
        raise ValueError("Invalid option. Choose either 'more' or 'less'.")

    updated_params = params.copy()
    updated_params["n_scientist_per_entry"] *= shift
    updated_params["n_professor_per_entry"] *= shift
    updated_params["n_engineer_per_entry"] *= shift
    updated_params["n_phd_per_entry"] *= shift

    updated_params_cf = params_cf.copy()
    updated_params_cf["n_scientist_per_entry"] *= shift
    updated_params_cf["n_professor_per_entry"] *= shift
    updated_params_cf["n_engineer_per_entry"] *= shift
    updated_params_cf["n_phd_per_entry"] *= shift

    return updated_params, updated_params_cf


def update_contender_seniority(params, params_cf, option):
    """
    Update the number of scientists and professors per entry, while adjusting
    engineers and PhDs in the opposite direction.

    Args:
        params : dict
            The main params dictionary.
        params_cf : dict
            The counterfactual params dictionary.
        option : str
            The option for updating contender seniority, either 'more' or 'less'.

    Returns
        updated_params : dict
            The updated main params dictionary.
        updated_params_cf : dict
            The updated counterfactual params dictionary.

    Warning messages:
        ValueError
            If the option provided is not 'more' or 'less'.
    """
    if option == "more":
        senior_shift = ar.small_shift
        junior_shift = 1 / ar.small_shift
    elif option == "less":
        senior_shift = 1 / ar.small_shift
        junior_shift = ar.small_shift
    else:
        raise ValueError("Invalid option. Choose either 'more' or 'less'.")

    updated_params = params.copy()
    updated_params["n_scientist_per_entry"] *= senior_shift
    updated_params["n_professor_per_entry"] *= senior_shift
    updated_params["n_engineer_per_entry"] *= junior_shift
    updated_params["n_phd_per_entry"] *= junior_shift

    updated_params_cf = params_cf.copy()
    updated_params_cf["n_scientist_per_entry"] *= senior_shift
    updated_params_cf["n_professor_per_entry"] *= senior_shift
    updated_params_cf["n_engineer_per_entry"] *= junior_shift
    updated_params_cf["n_phd_per_entry"] *= junior_shift

    return updated_params, updated_params_cf


def update_attendee_seniority(params, params_cf, option):
    if option == "more":
        senior_shift = ar.small_shift
        junior_shift = 1 / ar.small_shift
    elif option == "less":
        senior_shift = 1 / ar.small_shift
        junior_shift = ar.small_shift
    else:
        raise ValueError("Invalid option. Choose either 'more' or 'less'.")

    updated_params = params.copy()
    updated_params_cf = params_cf.copy()

    for key in ["fraction_attendee_scientist", "fraction_attendee_professor"]:
        updated_params[key] = manipulate_mean(updated_params, key, senior_shift)
        updated_params_cf[key] = manipulate_mean(updated_params_cf, key, senior_shift)

    key = "fraction_attendee_engineer"
    updated_params[key] = manipulate_mean(updated_params, key, junior_shift)
    updated_params_cf[key] = manipulate_mean(updated_params_cf, key, junior_shift)

    return updated_params, updated_params_cf


def update_hours_on_entry(params, params_cf, more_or_less):
    """
    Update hours spent on entry parameters for different contender types.

    Args:
        params : dict
            The main params dictionary.
        params_cf : dict
            The counterfactual params dictionary.
        more_or_less : str
            A string, either 'more' or 'less', determining whether to increase or
            decrease the hours spent on entry parameters.

    Returns:
        updated_params : dict
            The updated main params dictionary.
        updated_params_cf : dict
            The updated counterfactual params dictionary.
    """
    updated_params = params.copy()
    updated_params_cf = params_cf.copy()
    shift = ar.small_shift if more_or_less == "more" else 1 / ar.small_shift
    keys = [
        "hours_on_entry_per_contender_scientist",
        "hours_on_entry_per_contender_professor",
        "hours_on_entry_per_contender_engineer",
        "hours_on_entry_per_contender_phd",
    ]

    for key in keys:
        updated_params[key] = params[key] * shift
        updated_params_cf[key] = params_cf[key] * shift

    return updated_params, updated_params_cf


def perform_robustness_checks(params, checks=None):
    """
    Perform a series of robustness checks on the input parameters and return an updated dictionary.

    Args:
        params (dict): A dictionary containing existing parameter specifications. Must include keys
                       'mainline' and 'mainline_cf'.
        checks (list, optional): A list of robustness checks to perform. If not provided, all available
                                 checks will be performed.

    Returns:
        updated_params (dict): An updated dictionary containing the results of the performed
                               robustness checks.
    """

    # Define a dictionary of functions for the available robustness checks
    robustness_checks = {
        "larger_difference_in_scientist_equivalence": update_scientist_equivalence,
        "smaller_difference_in_scientist_equivalence": update_scientist_equivalence,
        "larger_labor_costs": update_labor,
        "smaller_labor_costs": update_labor,
        "larger_fixed_costs": update_fixed_cost_other,
        "smaller_fixed_costs": update_fixed_cost_other,
        "more_researchers_per_entry": update_number_per_entry,
        "fewer_researchers_per_entry": update_number_per_entry,
        "more_senior_contender_composition": update_contender_seniority,
        "less_senior_contender_composition": update_contender_seniority,
        "more_senior_attendee_composition": update_attendee_seniority,
        "less_senior_attendee_composition": update_attendee_seniority,
        "more_hours_on_each_entry": update_hours_on_entry,
        "fewer_hours_on_each_entry": update_hours_on_entry,
        "better_job_prospects": update_job_prospects,
        "worse_job_prospects": update_job_prospects,
        "better_talent_spotting": update_talent_spotting,
        "worse_talent_spotting": update_talent_spotting,
    }

    if checks is None:
        checks = list(robustness_checks.keys())

    # Create a copy of the input params dictionary to avoid modifying the original
    updated_params = params.copy()

    # Loop over the desired checks and call the corresponding functions
    for check in checks:
        if check in robustness_checks:
            func = robustness_checks[check]
            update_type = check.split("_")[0]
            updated_params[check], updated_params[check + "_cf"] = func(
                params["mainline"], params["mainline_cf"], update_type
            )

    return updated_params
