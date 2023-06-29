"""
Purpose: functions for QARYs over time
"""


"""
Imports
"""

import numpy as np  # for mean

import utilities.functions.research_relevance as aet
import utilities.functions.quality_adjustment as qa


"""
Helper functions
"""


def process_input_dicts(input_dict, researcher_type):
    """
    Purpose: process the input dictionary to remove researcher type suffixes from the keys

    Args:
        input_dict (dict): A dictionary containing various parameters, including 'years_since_phd' keys.
        researcher_type (str): A string specifying the researcher type (e.g. 'scientist', 'professor', 'engineer', 'phd')

    Returns:
        dict: An updated dictionary with researcher type suffixes removed from the keys.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if key.endswith(f"_{researcher_type}"):
            new_key = key[: -len(f"{researcher_type}")]
        else:
            new_key = key
        output_dict[new_key] = value
    return output_dict


def convert_years_since_to_until_phd(params_sampled):
    """
    Transforms 'years_since_phd' parameters in the input dictionary to the corresponding 'years_until_phd' parameters.

    Args:
        params_sampled (dict): A dictionary containing various parameters, including 'years_since_phd' keys.

    Returns:
        dict: An updated dictionary with 'years_until_phd' parameters calculated from 'years_since_phd' parameters.
    """
    # Create a copy of the input dictionary to prevent modifying the original dictionary
    updated_params = params_sampled.copy()

    # Iterate through the keys in the input dictionary
    for key, value in params_sampled.items():
        if key.startswith("years_since_phd"):
            # Check if there's a person type attached to the key
            if "_" in key[15:]:
                researcher_type = key[15:]
            else:
                researcher_type = ""

            years_since_phd = value
            years_in_phd = updated_params["years_in_phd"]
            years_until_phd_key = "years_until_phd" + researcher_type

            # Calculate years_until_phd and update the dictionary
            updated_params[years_until_phd_key] = -years_since_phd - years_in_phd
    return updated_params


def convert_hours_to_years_in_program(params_sampled):
    """
    Transforms 'hours_per_{participant}_{researcher type}' parameters in the input dictionary
    to the corresponding 'years_in_program_{participant}_{researcher type}' parameters.

    Args:
        params_sampled (dict): A dictionary containing various parameters,
            including 'hours_per_{participant}_{researcher type}' keys.

    Returns:
        dict: An updated dictionary with 'years_in_program_{participant}_{researcher type}'
            parameters calculated from 'hours_per_{participant}_{researcher type}' parameters.
    """
    # Create a copy of the input dictionary to prevent modifying the original dictionary
    updated_params = params_sampled.copy()

    # Iterate through the keys in the input dictionary
    for key, value in params_sampled.items():
        if key.startswith("hours_on_entry_per_"):
            researcher_participant_key = key[19:]

            hours_per_researcher_participant = value
            hours_scientist_per_year = updated_params["hours_scientist_per_year"]
            years_in_program_key = "years_in_program_" + researcher_participant_key

            # Calculate years_in_program and update the dictionary
            updated_params[years_in_program_key] = (
                hours_per_researcher_participant / hours_scientist_per_year
            )
    return updated_params


def update_research_relevance_multiplier(
    params_sampled, participant_type, researcher_type, undergrad_type
):
    """
    Adds 'research_relevance_multiplier_after_phd_{participant_type}_{researcher_type}_{undergrad_type}'
    keys in the input dictionary if they are not present and initializes them with a vector of 1s of the
    same length as the other vectors in the dictionary.

    Args:
        params_sampled (dict): A dictionary containing various parameters.
        participant_type (str): A string specifying the participant type (e.g. '_attendee', '_contender', '_student')
        researcher_type (str): A string specifying the researcher type (e.g. '_scientist', '_professor', '_engineer', '_phd', '_undergrad')
        undergrad_type (str): A string specifying the undergrad type (e.g. '_via_phd', '_not_via_phd')

    Returns:
        dict: An updated dictionary with 'research_relevance_multiplier_after_phd_{participant_type}_{researcher_type}_{undergrad_type}'
        keys added and initialized with 1s if they were not present.
    """
    # Create a copy of the input dictionary to prevent modifying the original dictionary
    updated_params = params_sampled.copy()

    # Get the length of the vectors in the dictionary. We assume that all vectors are the same length.
    vector_length = len(next(iter(params_sampled.values())))

    # Adjust these as per your undergrad types
    relevance_key = f"research_relevance_multiplier_after_phd{participant_type}{researcher_type}{undergrad_type}"

    # Check if the key is not present in the dictionary
    if relevance_key not in updated_params:
        # Add the key to the dictionary and initialize it with an array of 1s
        updated_params[relevance_key] = np.ones(vector_length)

    return updated_params


def compute_derivative(f, t, delta=1e-6):
    """
    Compute the first derivative of a function at a given point using the central difference method.

    Args:
        f (function): The function for which to compute the derivative.
        t (float): The point at which to compute the derivative.
        delta (float, optional): The step size for the central difference method. Default is 1e-6.

    Returns:
        float: The computed first derivative value.
    """
    return (f(t + delta) - f(t - delta)) / (2 * delta)


def create_variable_time_grid(
    f,
    t_min,
    t_max,
    n_points_initial=300,
    n_points_new=30,
    delta=1e-6,
    slope_threshold=0.1,
):
    """
    Create a variable time grid based on the magnitude of the first derivative of a function.
    Regions where the function changes rapidly (i.e., high slope magnitude) will have more points.

    Args:
        f (function): The function for which the time grid is created.
        t_min (float): The starting point of the time grid.
        t_max (float): The ending point of the time grid.
        n_points_initial (int): The number of initial points in the time grid.
        n_points_new (int): The number of new points to add in high slope regions.
        delta (float, optional): The interval for the central difference in the derivative calculation. Defaults to 1e-6.
        slope_threshold (float, optional): The absolute slope magnitude threshold for refining the time grid. Defaults to 0.5.

    Returns:
        numpy.ndarray: The variable time grid.
    """
    # Create an initial time grid
    t = np.linspace(t_min, t_max, n_points_initial)

    # Compute the absolute slopes at each point in the time grid
    slopes = np.abs([compute_derivative(f, ti, delta) for ti in t])

    # Identify the indices of the points where the slope exceeds the threshold
    high_slope_indices = np.where(slopes > slope_threshold)[0]

    # Initialize a list to hold the refined time grid points
    refined_t = list(t)

    # Add an arbitrary number of new points between each pair of original points in high slope regions
    for i in high_slope_indices:
        if i + 1 < len(t):  # Check to avoid out-of-bounds error
            t1, t2 = t[i], t[i + 1]
            step = (t2 - t1) / (n_points_new + 1)
            new_points = [t1 + step * k for k in range(1, n_points_new + 1)]
            refined_t.extend(new_points)

    # Sort the list of time grid points and convert it to a numpy array
    refined_t = np.sort(np.array(refined_t))
    return refined_t


"""
QARYs over time
"""


def benefit_qarys_over_time(
    params,
    params_sampled,
    derived_params_sampled,
    participant_type="_contender",
    researcher_type="",
    integration_max_time=60,
):
    """
    Calculates the value of research over time for a student who becomes a
    scientist-equivalent.

    Inputs:
        params (dict): Dictionary of parameters.
        params_sampled (dict): Dictionary of sampled parameters.
        derived_params_sampled (dict): Dictionary of new sampled parameters.
        researcher_type (str): String describing the person type (e.g., 'phd', 'engineer', 'scientist').
        participant_type (str): String describing the participant type (e.g., '_contender').

    Outputs:
        params (dict): Dictionary of parameters.
        params_sampled (dict): Dictionary of sampled parameters.
        derived_params_sampled (dict): Dictionary of new sampled parameters.
        derived_functions (dict): Dictionary of new functions.
    """

    # Prepare input dictionaries
    if "via_phd" in researcher_type:
        undergrad_type = researcher_type.replace("_undergrad", "")
        researcher_type = "_undergrad"
    else:
        undergrad_type = ""

    # Modify parameters
    params_sampled = convert_years_since_to_until_phd(params_sampled)
    params_sampled = convert_hours_to_years_in_program(params_sampled)
    params_sampled = update_research_relevance_multiplier(
        params_sampled, participant_type, researcher_type, undergrad_type
    )

    params_fns_over_t = {
        "research_relevance": params_sampled[
            "research_relevance" + participant_type + researcher_type + undergrad_type
        ],
        "research_relevance_multiplier_after_phd": params_sampled[
            "research_relevance_multiplier_after_phd"
            + participant_type
            + researcher_type
            + undergrad_type
        ],
        "research_relevance_during_program": params_sampled[
            "research_relevance_during_program"
            + participant_type
            + researcher_type
            + undergrad_type
        ],
        "years_in_program": params_sampled[
            "years_in_program" + participant_type + researcher_type
        ],
        "years_in_phd": params_sampled["years_in_phd"],
        "years_until_phd": params_sampled["years_until_phd" + researcher_type],
        "slope_productivity_life_cycle": params_sampled[
            "slope_productivity_life_cycle"
        ],
        "pivot_productivity_life_cycle": params_sampled[
            "pivot_productivity_life_cycle"
        ],
        "pivot_staying_in_ai": params_sampled["pivot_staying_in_ai"],
        "end_staying_in_ai": params_sampled["end_staying_in_ai"],
        "slope_staying_in_ai": params_sampled["slope_staying_in_ai"],
        "hours_scientist_per_year": params_sampled["hours_scientist_per_year"],
        "research_discount_rate": params_sampled["research_discount_rate"],
        "ability_at_first": params_sampled[
            "ability_at_first" + participant_type + researcher_type
        ],
        "ability_at_pivot": params_sampled[
            "ability_at_pivot" + participant_type + researcher_type
        ],
        "ability_pivot_point": params_sampled[
            "ability_pivot_point" + participant_type + researcher_type
        ],
    }

    if undergrad_type == "_not_via_phd":
        params_fns_over_t["years_in_phd"][:] = 0

    params_fns_over_t, all_inputs_modified = qa.process_vectors_input(
        **params_fns_over_t
    )

    # Define research avenue value over time
    def research_relevance_over_t(t):
        """Calculate research ITN over time."""
        research_relevance_over_t = aet.research_relevance_piecewise_over_t(
            params_fns_over_t["research_relevance"],
            params_fns_over_t["research_relevance_multiplier_after_phd"],
            years_in_phd=params_fns_over_t["years_in_phd"],
            years_until_phd=params_fns_over_t["years_until_phd"],
            years_in_program=params_fns_over_t["years_in_program"],
            research_relevance_during_program=params_fns_over_t[
                "research_relevance_during_program"
            ],
        )(t)
        return research_relevance_over_t

    # Calculate productivity life-cycle
    years_until_scientist = (
        params_fns_over_t["years_in_phd"] + params_fns_over_t["years_until_phd"]
    )

    def productivity_over_t(t):
        """Calculate productivity over time."""
        return qa.productivity_life_cycle(
            params_fns_over_t["slope_productivity_life_cycle"],
            params_fns_over_t["years_until_phd"] + 6,
            params_fns_over_t["pivot_productivity_life_cycle"],
        )(t)

    # Define functions for career end probability
    if params_fns_over_t["slope_staying_in_ai"] != 0:
        slope_staying_in_ai = params_fns_over_t["slope_staying_in_ai"] + (
            years_until_scientist / 100
        )
    else:
        slope_staying_in_ai = params_fns_over_t["slope_staying_in_ai"]
    pivot_staying_in_ai = params_fns_over_t["pivot_staying_in_ai"] - (
        years_until_scientist / 2
    )
    end_staying_in_ai = params_fns_over_t["end_staying_in_ai"] + years_until_scientist

    def p_staying_in_ai_over_t(t):
        """Calculate career end probability over time."""
        return qa.p_staying_in_ai_normalized(
            slope_staying_in_ai,
            years_until_scientist,
            pivot_staying_in_ai,
            end_staying_in_ai,
        )(t)

    # Define functions for hours worked
    def hours_over_t(t):
        """Calculate hours worked over time."""
        return qa.hours(
            params_fns_over_t["hours_scientist_per_year"],
            params_fns_over_t["years_until_phd"],
            100,
        )(t)

    # Define functions for research time discount
    def research_time_discount_over_t(t):
        """Calculate research time discount over time."""
        return qa.research_time_discounting(
            params_fns_over_t["research_discount_rate"]
        )(t)

    # Define functions for qarys over time
    def qarys_over_t(t):
        """Calculate qarys over time."""
        return (
            research_relevance_over_t(t)
            * productivity_over_t(t)
            * p_staying_in_ai_over_t(t)
            * hours_over_t(t)
            * research_time_discount_over_t(t)
            / params_fns_over_t["hours_scientist_per_year"]
        )

    # Mean ability
    try:
        n_participant = derived_params_sampled["n" + participant_type]
    except KeyError:
        n_participant = derived_params_sampled["n" + participant_type + researcher_type]

    mean_ability = qa.mean_ability_piecewise(
        n_participant,
        params_fns_over_t["ability_at_first"],
        params_fns_over_t["ability_at_pivot"],
        params_fns_over_t["ability_pivot_point"],
    )

    # Integrate over time and adjust for researcher ability
    t = create_variable_time_grid(qarys_over_t, 0, integration_max_time)
    qarys_values = np.array([qarys_over_t(ti) for ti in t])

    if researcher_type == "_phd" or undergrad_type == "_via_phd":
        phd_end_time = (
            params_fns_over_t["years_in_phd"] + params_fns_over_t["years_until_phd"]
        )

        t_during_phd = t[t <= phd_end_time]
        t_after_phd = t[t > phd_end_time]

        qarys_values_during_phd = qarys_values[t <= phd_end_time]
        qarys_values_after_phd = qarys_values[t > phd_end_time]

        if all_inputs_modified:
            qarys_per_during_phd = np.trapz(qarys_values_during_phd, t_during_phd)
            qarys_per_after_phd = np.trapz(qarys_values_after_phd, t_after_phd)

            qarys_per_during_phd = [qarys_per_during_phd] * len(
                params_sampled["years_in_phd"]
            )
            qarys_per_after_phd = [qarys_per_after_phd] * len(
                params_sampled["years_in_phd"]
            )
        else:
            qarys_per_during_phd = np.array(
                [
                    np.trapz(qarys_values_during_phd[:, i], t_during_phd)
                    for i in range(qarys_values_during_phd.shape[1])
                ]
            )
            qarys_per_after_phd = np.array(
                [
                    np.trapz(qarys_values_after_phd[:, i], t_after_phd)
                    for i in range(qarys_values_after_phd.shape[1])
                ]
            )

        qarys_per_during_phd = np.multiply(qarys_per_during_phd, mean_ability)
        qarys_per_after_phd = np.multiply(qarys_per_after_phd, mean_ability)
        qarys_per_phd = qarys_per_during_phd + qarys_per_after_phd

        # Output
        derived_params_sampled = {
            **derived_params_sampled,
            f"qarys_per_during{participant_type}{researcher_type}{undergrad_type}": qarys_per_during_phd,
            f"qarys_per_after{participant_type}{researcher_type}{undergrad_type}": qarys_per_after_phd,
            f"qarys_per{participant_type}{researcher_type}{undergrad_type}": qarys_per_phd,
            f"mean_ability{participant_type}{researcher_type}{undergrad_type}": mean_ability,
        }
    else:
        if all_inputs_modified:
            qarys_per = np.trapz(qarys_values, t)
            # Convert back to vector
            qarys_per = [qarys_per] * len(params_sampled["years_in_phd"])
        else:
            # Compute element-wise integral values using numpy.trapz
            qarys_per = np.array(
                [np.trapz(qarys_values[:, i], t) for i in range(qarys_values.shape[1])]
            )

        qarys_per = np.multiply(qarys_per, mean_ability)

        # Output
        derived_params_sampled = {
            **derived_params_sampled,
            f"qarys_per{participant_type}{researcher_type}{undergrad_type}": qarys_per,
            f"mean_ability{participant_type}{researcher_type}{undergrad_type}": mean_ability,
        }

    # Output
    derived_functions = {
        f"research_relevance_over_t{participant_type}{researcher_type}{undergrad_type}": research_relevance_over_t,
        f"productivity_over_t{participant_type}{researcher_type}{undergrad_type}": productivity_over_t,
        f"p_staying_in_ai_over_t{participant_type}{researcher_type}{undergrad_type}": p_staying_in_ai_over_t,
        f"hours_over_t{participant_type}{researcher_type}{undergrad_type}": hours_over_t,
        f"research_time_discount_over_t{participant_type}{researcher_type}{undergrad_type}": research_time_discount_over_t,
        f"qarys_over_t{participant_type}{researcher_type}{undergrad_type}": qarys_over_t,
    }

    return params, params_sampled, derived_params_sampled, derived_functions


def combine_dicts_across_researcher_types(
    param_dicts_list, func_dicts_list, participant="contender", researcher_types=None
):
    """
    Combine the parameter and function dictionaries for different researcher types.

    Inputs:
        param_dicts_list: list of parameter dictionaries for different researcher types
        func_dicts_list: list of function dictionaries for different researcher types
        participant: a string specifying the participant type (default: 'contender')
        researcher_types: list of researcher types (e.g. ['scientist', 'professor', 'engineer', 'phd'])

    Returns:
        combined_params_dict: the combined parameter dictionary
        combined_funcs_dict: the combined function dictionary
    """

    if researcher_types is None:
        researcher_types = ["scientist", "professor", "engineer", "phd"]

    # Combine parameter dictionaries
    combined_params_dict = {
        k: v
        for d in param_dicts_list
        for k, v in d.items()
        if k.startswith("qarys_per") or k.startswith("mean_ability")
    }

    # Add the common keys from the first dictionary in the list
    for k, v in param_dicts_list[0].items():
        if k not in combined_params_dict:
            combined_params_dict[k] = v

    # Combine function dictionaries
    combined_funcs_dict = {k: v for d in func_dicts_list for k, v in d.items()}

    # Calculate QARYs by category
    for rt in researcher_types:
        if rt == "phd" or rt == "undergrad_via_phd":
            key_n_during = f"n_scientist_equivalent_{participant}_{rt}_during"
            key_n_after = f"n_scientist_equivalent_{participant}_{rt}_after"
            key_qarys_during = f"qarys_per_during_{participant}_{rt}"
            key_qarys_after = f"qarys_per_after_{participant}_{rt}"
            combined_params_dict[f"qarys_{participant}_{rt}_during"] = (
                combined_params_dict[key_n_during]
                * combined_params_dict[key_qarys_during]
            )
            combined_params_dict[f"qarys_{participant}_{rt}_after"] = (
                combined_params_dict[key_n_after]
                * combined_params_dict[key_qarys_after]
            )
            combined_params_dict[f"qarys_{participant}_{rt}"] = (
                combined_params_dict[f"qarys_{participant}_{rt}_during"]
                + combined_params_dict[f"qarys_{participant}_{rt}_after"]
            )
        else:
            key_n = f"n_scientist_equivalent_{participant}_{rt}"
            key_qarys = f"qarys_per_{participant}_{rt}"
            combined_params_dict[f"qarys_{participant}_{rt}"] = (
                combined_params_dict[key_n] * combined_params_dict[key_qarys]
            )

    # Calculate overall QARYs
    combined_params_dict[f"qarys_{participant}"] = sum(
        combined_params_dict[f"qarys_{participant}_{rt}"] for rt in researcher_types
    )

    return combined_params_dict, combined_funcs_dict


def combine_researcher_dicts_across_participants(
    derived_params_sampled_list, functions_list
):
    """
    Combines dictionaries of derived parameters and functions for multiple participants.

    Args:
        params_sampled_list: A list of dictionaries containing derived parameters for each participant.
        functions_list: A list of dictionaries containing derived functions for each participant.

    Returns:
        A tuple containing the combined dictionaries for derived parameters and functions.
    """

    # Combine derived_params_sampled dictionaries
    combined_derived_params_sampled = derived_params_sampled_list[0].copy()
    for derived_params_sampled in derived_params_sampled_list[1:]:
        for k, v in derived_params_sampled.items():
            if k not in combined_derived_params_sampled:
                combined_derived_params_sampled[k] = v

    # Combine derived_functions dictionaries
    combined_functions = {}
    for functions in functions_list:
        combined_functions.update(functions)

    return combined_derived_params_sampled, combined_functions
