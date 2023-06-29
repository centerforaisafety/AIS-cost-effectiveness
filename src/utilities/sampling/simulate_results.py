"""
Purpose: use models to simulate results
"""


"""
Imports
"""

import sys

sys.path.append("src")

# Common python packages
import pandas as pd  # for reading csv
import numpy as np  # for sorting arrays


"""
Functions
"""


def get_program_data(
    programs,
    default_parameters,
    master_functions,
    n_sim,
    time_points,
    parameter_specification="mainline",
):
    """
    Computes data for a set of programs given their default parameters and master functions.

    Args:
        programs (list): A list of program names.
        default_parameters (dict): A dictionary where keys are program names and values are default parameters for each program.
        master_functions (dict): A dictionary where keys are program names and values are master functions for each program.
        n_sim (int): The number of simulations to run for each program.
        time_points (list): A list of time points at which to compute the data.
        parameter_specification (str, optional): The name of the parameter specification to use. Default is "mainline".

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary has program names as keys and dataframes of function values as values. The second dictionary has program names as keys and dataframes of parameters as values.
    """
    df_functions = {}
    df_params = {}

    for i in programs:
        print("Computing data for program: ", i)

        # Check if the parameter specification is in the params dictionary for this program
        try:
            if parameter_specification not in default_parameters[i].params:
                continue
        except:
            try:
                if parameter_specification not in default_parameters[i]:
                    continue
            except:
                try:
                    if parameter_specification not in default_parameters:
                        continue
                except:
                    pass

        # get master function for this program
        master_function = master_functions[i].mfn

        # get parameters for this program and specification
        try:
            params = default_parameters[i].params[parameter_specification]
            params_cf = default_parameters[i].params[parameter_specification + "_cf"]
        except:
            try:
                params = default_parameters[i][parameter_specification]
                params_cf = default_parameters[i][parameter_specification + "_cf"]
            except:
                params = default_parameters[parameter_specification]
                params_cf = default_parameters[parameter_specification + "_cf"]

        # run model for parameters
        (
            params,
            params_sampled,
            derived_params_sampled,
            derived_functions,
        ) = master_function(params, params_cf, n_sim)

        # convert to dataframes
        params_sampled = pd.DataFrame(params_sampled)
        derived_params_sampled = pd.DataFrame(derived_params_sampled)
        df_params[i] = pd.concat([params_sampled, derived_params_sampled], axis=1)

        output_matrices = {}

        for function_name, function in derived_functions.items():
            output_list = [function(t) for t in time_points]
            output_matrix = np.stack(output_list, axis=0)
            output_matrices[function_name] = output_matrix

        if len(next(iter(output_matrices.values()))) == len(time_points):
            # If length of output_matrices dictionary values is same as number
            # of time_points
            output_data = {"time_point": time_points}
        else:
            output_data = {
                "time_point": np.repeat(time_points, len(range(0, n_sim))),
                "element": np.tile(range(0, n_sim), len(time_points)),
            }

        for function_name, output_matrix in output_matrices.items():
            output_data[function_name] = output_matrix.flatten()

        # Convert the output_data dictionary to a DataFrame
        df_functions[i] = pd.DataFrame(output_data)

    return df_functions, df_params


def get_scenario_data(
    programs,
    default_parameters,
    master_functions,
    n_sim,
    time_points,
    scenarios=["mainline"],
    create_df_functions=True,
    share_params_cf=True,
):
    """
    Computes data for a set of programs across a range of scenarios

    Args:
        programs (list): A list of program names.
        default_parameters (dict): A dictionary where keys are program names and values are default parameters for each program.
        master_functions (dict): A dictionary where keys are program names and values are master functions for each program.
        n_sim (int): The number of simulations to run for each program.
        time_points (list): A list of time points at which to compute the data.
        scenarios (list): A list of scenarios to compute data for.
        create_df_functions (bool, optional): Whether to create a dataframe of function values. Default is True.
        share_params_cf (bool, optional): Whether to share the parameters across scenarios. Default is True.

    Returns:
        df_params (dict): A dictionary of simulated parameters for each program and scenario.
        df_functions (dict): Optional, depending on create_df_functions flag; a dictionary of values of functions across programs and scenarios.
    """
    df_functions = {} if create_df_functions else None
    df_params = {}

    default_parameters = default_parameters.copy()

    for i in programs:
        # get master function for this program
        master_function = master_functions[i].mfn

        # Initialize dataframes for this program
        if create_df_functions:
            df_functions[i] = {}
        df_params[i] = {}

        for scenario in scenarios:
            print("Computing data for program ", i, " and scenario ", scenario)
            # Check if the scenario is in the params dictionary for this program
            if scenario not in default_parameters[i].params:
                continue

            # get default parameters for this program and scenario
            params = default_parameters[i].params[scenario]
            if share_params_cf:
                params_cf = default_parameters[i].params[scenarios[0] + "_cf"]
            else:
                params_cf = default_parameters[i].params[scenario + "_cf"]

            # run model for default parameters
            (
                params,
                params_sampled,
                derived_params_sampled,
                derived_functions,
            ) = master_function(params, params_cf, n_sim)

            # convert to dataframes
            params_sampled = pd.DataFrame(params_sampled)
            derived_params_sampled = pd.DataFrame(derived_params_sampled)
            df_params[i][scenario] = pd.concat(
                [params_sampled, derived_params_sampled], axis=1
            )

            if create_df_functions:
                output_matrices = {}

                for function_name, function in derived_functions.items():
                    output_list = [function(t) for t in time_points]
                    output_matrix = np.stack(output_list, axis=0)
                    output_matrices[function_name] = output_matrix

                if len(output_matrices["qarys_over_t"]) == len(time_points):
                    output_data = {"time_point": time_points}
                else:
                    output_data = {
                        "time_point": np.repeat(time_points, len(range(0, n_sim))),
                        "element": np.tile(range(0, n_sim), len(time_points)),
                    }

                for function_name, output_matrix in output_matrices.items():
                    output_data[function_name] = output_matrix.flatten()

                # Convert the output_data dictionary to a DataFrame
                df_functions[i][scenario] = pd.DataFrame(output_data)

                return df_functions, df_params

    return df_params
