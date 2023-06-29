"""
Purpose: get outcomes for many different possible levels of budget
"""


"""
Imports
"""

import sys

sys.path.append("src")

# Sampling
import utilities.sampling.simulate_results as so

# Common python packages
import pandas as pd  # for reading csv

# Simulations
from squigglepy.numbers import K, M


"""
Functions
"""


def get_budget_sweep_data(
    programs,
    budget_values,
    default_parameters,
    master_functions,
    n_sim,
    time_points,
    estimate_participants=False,
):
    """
    Computes outcomes for a set of programs over different budget values.

    Inputs:
        programs (list): A list of program names.
        budget_values (list): A list of budget values to use.
        default_parameters (dict): A dictionary where keys are program names and values are default parameters for each program.
        master_functions (dict): A dictionary where keys are program names and values are master functions for each program.
        n_sim (int): The number of simulations to run for each program.
        time_points (list): A list of time points at which to compute the data.
        estimate_participants (bool, optional): Whether to estimate the average number of participants. Default is False.

    Returns:
        output_df (pd.DataFrame): A dataframe containing outcomes for different budget values.
    """
    output_data = []

    for budget in budget_values:
        # Print the current budget value
        print("Computing outcomes for budget level: ", int(budget))

        # Update the budget parameter for each program
        for i in programs:
            default_parameters[i].params["mainline"]["target_budget"] = budget
            default_parameters[i].params["mainline_cf"]["target_budget"] = budget

        # Get the program data
        df_functions, df_params = so.get_program_data(
            programs, default_parameters, master_functions, n_sim, time_points
        )

        # Calculate the mean of qarys and qarys_cf for each program
        for i in programs:
            mean_qarys = df_params[i]["qarys"].mean()
            mean_qarys_cf = df_params[i]["qarys_cf"].mean()

            # Estimate the average number of participants
            if estimate_participants:
                # Determine which participant-related columns are present in the data
                participant_types = [
                    "n_student",
                    "n_attendee",
                    "n_contender",
                    "n_student_undergrad",
                    "n_student_phd",
                ]
                mean_participants = {
                    ptype: df_params[i][ptype].mean()
                    if ptype in df_params[i].columns
                    else None
                    for ptype in participant_types
                }
            else:
                mean_participants = None

            output_dict = {
                "target_budget": budget,
                "program": i,
                "mean_qarys": mean_qarys,
                "mean_qarys_cf": mean_qarys_cf,
            }
            output_dict.update(mean_participants)
            output_data.append(output_dict)

    # Convert the output_data list to a DataFrame
    output_df = pd.DataFrame(output_data)

    # Calculate the difference between mean_qarys and mean_qarys_cf
    output_df["mean_qarys_diff"] = output_df["mean_qarys"] - output_df["mean_qarys_cf"]

    # Calculate the cost-effectiveness
    output_df["cost_effectiveness"] = output_df["mean_qarys_diff"] / (
        output_df["target_budget"] / (1 * M)
    )
    output_df.loc[
        output_df["target_budget"] == 0, "cost_effectiveness"
    ] = 0  # Handle division by zero

    return output_df
