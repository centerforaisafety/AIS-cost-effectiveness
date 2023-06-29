"""
Purpose: functions for quality adjustment, excluding research relevance
"""


"""
Imports
"""

import numpy as np  # for mean


"""
Helper functions
"""


def process_vector_input(input_value):
    """
    Process identical vector inputs
        If vector inputs are repeated scalars, convert to scalar so they can be
        processed more cheaply.

        This greatly speeds up integration of functions over time in cases where we
        do not specify uncertainty over their parameters.

    Inputs:
        input_value: input value to be processed

    Returns:
        tuple: processed input value and a boolean indicating if input was modified
    """
    if isinstance(input_value, tuple) and len(input_value) == 1:
        input_value = input_value[0]
        return input_value, True

    if isinstance(input_value, np.ndarray) and np.all(input_value == input_value[0]):
        input_value = input_value[0]
        return input_value, True

    return input_value, False


def process_vectors_input(**kwargs):
    """
    Process input parameters and return modified or unmodified values depending on whether all inputs were modified or not.

    Inputs:
        **kwargs: arbitrary number of input parameters to be processed

    Returns:
        tuple: a dictionary containing processed input parameters and a boolean indicating if all input parameters were modified
    """
    inputs_modified = {k: False for k in kwargs.keys()}
    processed_inputs = {}

    for k, v in kwargs.items():
        processed_input, input_modified = process_vector_input(v)
        inputs_modified[k] = input_modified
        processed_inputs[k] = processed_input

    all_inputs_modified = all(inputs_modified.values())

    if all_inputs_modified:
        return processed_inputs, True
    else:
        return kwargs, False


"""
Research discounting over time 
"""


# Research discounting over time
def research_time_discounting(discount_rate):
    """
    Research discounting over time.

    Inputs:
        discount rate

    Outputs:
        function of time
    """

    def f(t):
        # geometric discounting
        return (1 - discount_rate) ** t

    return f


"""
Productivity over time
"""


def productivity_life_cycle(slope, start, pivot):
    """
    Returns a function that describes the productivity life-cycle of a scientist.

    Inputs:
        slope (float): The slope of the sigmoid function that describes the productivity life-cycle.
        start (float): The start year of the productivity life-cycle.
        pivot (float): The pivot year of the productivity life-cycle.
        years_until_phd (float): The number of years until the scientist gets their PhD.

    Returns:
        function: A function that takes a time t and returns the productivity at that time.
    """

    def f(t):
        # If slope is 0, the function always returns 1
        if slope == 0:
            return 1
        # life cycle
        life_cycle_t = 1 / (1 + np.exp(slope * (t - (start + pivot))))
        # output
        # return np.where((t < years_until_phd) | (t < start), 0, life_cycle_t)
        return np.where((t < 0), 0, life_cycle_t)

    return f


"""
Probability of remaining in AI profession over time
"""


def p_staying_in_ai(slope, start, pivot):
    """
    Probability of career ending for research scientist-equivalent person.

    Inputs:
        slope (float): The slope of the sigmoid function.
        start (float): The start time.
        pivot (float): The pivot point of the sigmoid function.

    Returns:
        function: A function of time representing the probability of career ending.
    """

    def f(t):
        p_staying_in_ai_t = 1 - (1 / (1 + np.exp(slope * (t - (start + pivot)))))
        return np.where(t < start, 0, p_staying_in_ai_t)

    return f


def p_staying_in_ai_normalized(slope, start, pivot, end):
    """
    Probability of career ending for student, normalized.

    Inputs:
        slope (float): The slope of the sigmoid function.
        start (float): The start time.
        pivot (float): The pivot point of the sigmoid function.
        end (float): The end time.

    Returns:
        function: A function of time representing the normalized probability of career ending.
    """
    p_staying_in_ai_at_start = p_staying_in_ai(slope, start, pivot)(max(start, 0))
    normalization_constant = 1 / p_staying_in_ai_at_start

    def f(t):
        constant_start_t = np.where(t < start, 1, 0)
        p_staying_in_ai_normalized_t = normalization_constant * p_staying_in_ai(
            slope, start, pivot
        )(t)
        p_staying_in_ai_normalized_with_constant_start_t = (
            constant_start_t + p_staying_in_ai_normalized_t
        )

        return np.where(t > end, 0, p_staying_in_ai_normalized_with_constant_start_t)

    return f


"""
Hours worked over time
"""


def hours(hours_per_year, years_until_phd, end):
    """
    Hours worked over time.

    Inputs:
        hours_per_year, years_until_phd, end

    Returns:
        function of time

    """

    start = np.max([0, years_until_phd])

    def f(t):
        # hours worked over life cycle (per-year)
        return np.where((t < start) | (t > end), 0, hours_per_year)

    return f


"""
Ability
"""


def calculate_constant_ability(
    best_ability, pivot_rank, average_ability, total_students
):
    """
    Calculates the constant ability level for ranks beyond a certain pivot point.

    Inputs:
        best_ability (float): The ability of the best student.
        pivot_rank (int): The rank of the pivot student.
        average_ability (float): The average ability across all students.
        total_students (int): The total number of students.

    Returns:
        float: The constant ability level for ranks beyond the pivot point.
    """
    # Calculate the sum of abilities for ranks up to the pivot point
    sum_abilities_up_to_pivot = (
        (best_ability + best_ability - (pivot_rank - 1)) * pivot_rank / 2
    )

    # Set up the equation for the average ability and solve for the constant ability level
    constant_ability = (
        average_ability * total_students - sum_abilities_up_to_pivot
    ) / (total_students - pivot_rank)

    return constant_ability


def piecewise_function(x, ability_at_first, ability_at_pivot, ability_pivot_point):
    """
    Piecewise function to calculate ability based on student rank.

    Inputs:
        x (int): Student rank.
        ability_at_first (float): Ability at first student.
        ability_at_pivot (float): Ability at pivot student.
        ability_pivot_point (int): Pivot person.

    Returns:
        float: Ability.
    """
    slope = (ability_at_pivot - ability_at_first) / (ability_pivot_point - 1)
    intercept = ability_at_first - slope

    if x < ability_pivot_point:
        return slope * x + intercept
    else:
        return ability_at_pivot


def mean_ability_piecewise(
    n_contender, ability_at_first, ability_at_pivot, ability_pivot_point
):
    """
    Compute the mean ability based on the given parameters.

    Inputs:
        n_contender: The total number of contenders (scalar or vector).
        ability_at_first: Ability at the first contender.
        ability_at_pivot: Ability at the pivot contender.
        ability_pivot_point: The pivot contender index.

    Returns:
        The mean ability as a scalar or a vector depending on the input n_contender.
    """
    if ability_at_first == ability_at_pivot:
        if np.isscalar(n_contender):
            return ability_at_pivot
        else:
            return np.full(len(n_contender), ability_at_pivot)

    # Calculate the slope and intercept for the first piece of the piecewise function
    slope = (ability_at_pivot - ability_at_first) / (ability_pivot_point - 1)
    intercept = ability_at_first - slope

    # Compute the mean ability
    if np.isscalar(n_contender):
        if n_contender < ability_pivot_point:
            mean_ability = (
                piecewise_function(
                    1, ability_at_first, ability_at_pivot, ability_pivot_point
                )
                + piecewise_function(
                    n_contender, ability_at_first, ability_at_pivot, ability_pivot_point
                )
            ) / 2
        else:  # n_contender >= ability_pivot_point
            mean_ability_pre_pivot = (
                piecewise_function(
                    1, ability_at_first, ability_at_pivot, ability_pivot_point
                )
                + piecewise_function(
                    ability_pivot_point,
                    ability_at_first,
                    ability_at_pivot,
                    ability_pivot_point,
                )
            ) / 2
            mean_ability = (
                mean_ability_pre_pivot * ability_pivot_point
                + ability_at_pivot * (n_contender - ability_pivot_point)
            ) / n_contender
        return mean_ability
    else:
        mean_ability_vector = np.zeros(len(n_contender))
        for idx, n in enumerate(n_contender):
            if n < ability_pivot_point:
                mean_ability_vector[idx] = (
                    piecewise_function(
                        1, ability_at_first, ability_at_pivot, ability_pivot_point
                    )
                    + piecewise_function(
                        n, ability_at_first, ability_at_pivot, ability_pivot_point
                    )
                ) / 2
            else:
                mean_ability_pre_pivot = (
                    piecewise_function(
                        1, ability_at_first, ability_at_pivot, ability_pivot_point
                    )
                    + piecewise_function(
                        ability_pivot_point,
                        ability_at_first,
                        ability_at_pivot,
                        ability_pivot_point,
                    )
                ) / 2
                mean_ability_vector[idx] = (
                    mean_ability_pre_pivot * ability_pivot_point
                    + ability_at_pivot * (n - ability_pivot_point)
                ) / n
        return mean_ability_vector
