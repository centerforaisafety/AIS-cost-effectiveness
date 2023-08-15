"""
Purpose: functions for number of program participants given variable cost
"""


"""
Imports
"""

import squigglepy as sq
import numpy as np


"""
Functions
"""


def expected_entries(expected_utility, slope=0.3, intercept=10):
    """
    Expected entries for a contest given expected utility.

    Args:
        expected_utility (float): Expected utility.
        slope (float): Scaling parameter.

    Returns:
        float: Expected number of entries.
    """
    return slope * expected_utility + intercept


def crra_utility(c, gamma=1):
    """
    CRRA utility function.

    Args:
        c (float): Consumption.
        gamma (float): Risk aversion parameter.

    Returns:
        float: Utility.
    """
    c = np.array(c)  # Convert list to numpy array if it isn't already
    if gamma == 1:
        return np.log(c)
    else:
        return (c ** (1 - gamma)) / (1 - gamma)


def poisson_mean_entries(expected_award, gamma=0.1, slope=0.3, intercept=10):
    """
    Computes the mean number of entries for a Poisson distribution.

    Args:
        expected_award (np.array): Expected award.
        slope (float): Scaling parameter for expected entries.
        gamma (float): Risk aversion parameter for CRRA utility.
        intercept (float): Intercept parameter for expected entries.

    Returns:
        float: Mean number of entries.
    """
    utility = crra_utility(expected_award, gamma)
    mean_entries = expected_entries(utility, slope, intercept)

    return mean_entries


def compute_n_entry(
    variable_cost,
    gamma=0.4,
    slope=0.7,
    intercept=10,
    initial_divisor=1000,
    threshold=1,
    verbose=False,
):
    """
    Compute the number of entries to a prize contest, given the costs of the awards,
    and some parameters for the utility function and the Poisson distribution.
    It iteratively finds the fixed point where the number of entries stabilizes.

    Args:
        variable_cost (np.array): The costs of the awards.
        slope (float, optional): The parameter for the expected_entries function. Defaults to 0.3.
        gamma (float, optional): The parameter for the utility function. Defaults to 0.1.
        initial_divisor (int, optional): The divisor used to calculate the initial number of entries. Defaults to 1000.
        threshold (float, optional): The threshold for convergence. Defaults to 1.
        verbose (bool, optional): Whether to print progress messages. Defaults to False.

    Returns:
        np.array: The computed number of entries for each award.
    """

    # Set the iterator
    iteration = 0

    # Calculate initial value for number of entries
    with np.errstate(divide="ignore", invalid="ignore"):
        n_entry = variable_cost / initial_divisor
        n_entry = np.nan_to_num(n_entry, nan=0)

    # Loop until convergence
    while True:
        previous_n_entry = n_entry

        # Calculate the new expected awards
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_award = np.where(n_entry == 0, 0, variable_cost / n_entry)
            expected_award = np.nan_to_num(expected_award, nan=0)

        # Calculate the mean number of entries
        n_entry = poisson_mean_entries(expected_award, gamma, slope, intercept)

        # Check if the change is less than the threshold
        if np.all(np.abs(previous_n_entry - n_entry) < threshold):
            break

        if verbose and iteration % 10 == 0:
            print(f"Iteration: {iteration}")
            print(
                f"Mean absolute change in number of entries: {np.mean(np.abs(previous_n_entry - n_entry))}"
            )

        iteration += 1

    # Now that the expected awards and mean entries have converged, you can add uncertainty
    n_entry = np.array([sq.poisson(mean) @ 1 for mean in n_entry])

    return n_entry


def compute_n_attendee(variable_cost, max_capacity, gamma=0.3, slope=0.2, intercept=0):
    """
    Compute the number of attendees of events.

    Args:
        variable_cost (np.array): Variable costs.
        max_capacity (int): Maximum capacity of the program.
        slope (float): Scaling parameter for expected entries.
        gamma (float): Risk aversion parameter for CRRA utility.
        intercept (float): Intercept parameter for expected entries.
        
    Returns:
        float: Number of attendees.
    """
    n_attendee = poisson_mean_entries(variable_cost, gamma, slope, intercept)

    # Add uncertainty around
    n_attendee = np.array([sq.poisson(mean) @ 1 for mean in n_attendee])

    # Truncate by max capacity
    n_attendee = np.minimum(n_attendee, max_capacity)

    return n_attendee


def compute_n_student(
    variable_cost, gamma=0.7, slope=1.4, intercept=20, n_students_deterministic=True
):
    """
    Compute the number of participants in a student program.

    Args:
        variable_cost (float): Variable cost.
        slope (float): Scaling parameter for expected entries.
        gamma (float): Risk aversion parameter for CRRA utility.

    Returns:
        float: Number of participants.
    """
    n_participant = poisson_mean_entries(variable_cost, gamma, slope, intercept)

    # If stochastic, add uncertainty around mean
    if not n_students_deterministic:
        n_participant = np.array([sq.poisson(mean) @ 1 for mean in n_participant])

    return n_participant
