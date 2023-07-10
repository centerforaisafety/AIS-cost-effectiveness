"""
Purpose: helper functions for plotting and tables
"""


"""
Imports
"""


import sys

sys.path.append("src")

# Common python packages
import re  # for regular expressions
import pandas as pd  # for reading csv
import numpy as np  # for sorting arrays
from tabulate import tabulate  # for printing tables


"""
Compare one variable from each program
"""


def find_shared_variables(data_dict, programs, selected_variables=None):
    """
    Find variables that are shared across programs

    Args:
        data_dict (dict): dictionary of dataframes
        programs (list): list of programs
        selected_variables (list): list of variables to consider

    Returns:
        varying_variables: list of shared variables
    """
    shared_variables = set(data_dict[programs[0]].columns)

    for program in programs[1:]:
        shared_variables.intersection_update(set(data_dict[program].columns))

    if selected_variables:
        shared_variables.intersection_update(set(selected_variables))

    varying_variables = []
    for variable in shared_variables:
        is_identical = False
        for program in programs:
            if data_dict[program][variable].nunique() > 1:
                is_identical = True
                break
        if is_identical:
            varying_variables.append(variable)

    return varying_variables


def generate_yticks(ymin, ymax):
    """
    Generate yticks for log scale

    Args:
        ymin (float): minimum value
        ymax (float): maximum value

    Returns:
        list: list of yticks
    """
    if ymin < 0 and ymax < 0:
        ymin_exp = -np.ceil(np.log10(-ymax))
        ymax_exp = -np.floor(np.log10(-ymin))
    elif ymin < 0:
        ymin_exp = -np.ceil(np.log10(-ymin))
        ymax_exp = np.ceil(np.log10(ymax))
    else:
        ymin_exp = np.floor(np.log10(ymin))
        ymax_exp = np.ceil(np.log10(ymax))

    num_ticks = int(ymax_exp - ymin_exp) + 1

    if num_ticks < 2:
        yticks = [ymin, ymax]
    else:
        if ymin < 0:
            pos_ticks = np.logspace(0, ymax_exp, num=int(ymax_exp) + 1)
            neg_ticks = -np.logspace(0, abs(ymin_exp), num=int(abs(ymin_exp)) + 1)[::-1]
            yticks = np.concatenate((neg_ticks, [0], pos_ticks))
        else:
            yticks = np.logspace(ymin_exp, ymax_exp, num=num_ticks)

    return yticks


def generate_xticks(xmin, xmax):
    """
    Generate xticks for log scale

    Args:
        xmin (float): minimum value
        xmax (float): maximum value
        num_ticks (int): number of ticks
        xticks (list): list of xticks

    Returns:
        list: list of xticks
    """
    xmin_exp = np.floor(np.log10(xmin))
    xmax_exp = np.ceil(np.log10(xmax))
    num_ticks = int(np.floor(xmax_exp - xmin_exp)) + 1
    xticks = np.logspace(xmin_exp, xmax_exp, num=num_ticks)
    return xticks


def prettify_label(
    label,
    words_to_capitalize=["cais", "mlss", "ais", "ai", "ml", "tdc"],
    words_to_remove=None,
    capitalize_each_word=False,
    expand_abbreviations=False,
    verbose=False,
):
    """
    Prettify a label by applying specified capitalization rules.
    This function replaces underscores with spaces, capitalizes specified words,
    converts certain abbreviations to their full forms, and optionally capitalizes every word.

    Args:
        label (str): The label to be prettified.
        words_to_capitalize (list, optional): A list of words to be capitalized. Defaults to None.
        words_to_remove (list, optional): A list of words to be removed. Defaults to None.
        capitalize_each_word (bool, optional): If True, capitalizes every word. Defaults to False.

    Returns:
        str: The prettified label.
    """
    words_to_capitalize = [word.lower() for word in words_to_capitalize]

    if words_to_remove is not None:
        for word in words_to_remove:
            label = label.replace(word, "")

    words = label.split("_")

    words_capitalized = [
        word.upper()
        if word.lower() in words_to_capitalize
        else word.capitalize()
        if i == 0 or capitalize_each_word
        else word
        for i, word in enumerate(words)
    ]

    label = "_".join(words_capitalized)

    label = re.sub(r"\bActual_", "", label)
    label = re.sub(r"P_scientist_equivalent", "Scientist-equivalence", label)
    label = re.sub(r"\bP_", "Probability of_", label)
    label = re.sub(r"\bN_", "Number of_", label)
    label = re.sub(r"pursue_AIS", "seriously pursuing AIS", label)
    label = re.sub(r"Staying_In_", "staying in_", label)
    label = re.sub(r"scientist_equivalent_student_", "scientist-equivalent_", label)
    label = re.sub(r"\bStudent_undergrad_", "Undergrad_", label)
    label = re.sub(r"Undergrad_Via_Phd", "pre-PhD students who do PhDs", label)
    label = re.sub(
        r"Undergrad_not_Via_Phd", "pre-PhD students who do not do PhDs", label
    )
    label = re.sub(r"Undergrad_via_phd", "Pre-PhD students who do PhDs", label)
    label = re.sub(
        r"Undergrad_not_via_phd", "Pre-PhD students who do not do PhDs", label
    )
    label = re.sub(r"Student_Phd_During", "PhD students during PhDs", label)
    label = re.sub(r"Student_Phd_After", "PhD students after PhDs", label)
    label = re.sub(r"Student_Undergrad_During", "pre-PhD students during PhDs", label)
    label = re.sub(r"Student_Undergrad_After", "pre-PhD students after PhDs", label)
    label = re.sub(r"phd_during", "PhD students during PhDs", label)
    label = re.sub(r"phd_after", "PhD students after PhDs", label)
    label = re.sub(r"undergrad_during", "pre-PhD students during PhDs", label)
    label = re.sub(r"undergrad_after", "pre-PhD students after PhDs", label)
    label = re.sub(r"Qarys", "QARYs", label)
    label = re.sub(r"Neurips", "NeurIPS", label)
    label = re.sub(r"Research_Value", "Research avenue relevance", label)
    label = re.sub(r"Research_value", "Research avenue relevance", label)
    label = re.sub(r"of_phd_given_", "of doing a PhD given_", label)
    label = re.sub(r"given_phd", "given do PhD", label)
    label = re.sub(r"given_not_phd", "given do not do PhD", label)
    label = re.sub(r"of_Attendee", "of event attendees", label)
    label = re.sub(r"of_Contender", "of award contenders", label)
    label = re.sub(r"Student_Phd", "PhD students", label)
    label = re.sub(r"Student_Undergrad", "Pre-PhD students", label)
    label = re.sub(r"Phd", "PhD students", label)
    label = re.sub(r"phd", "PhD students", label)
    label = re.sub(r"\bUndergrad\b", "Pre-PhD students", label)

    pretty_words = label.split("_")

    if expand_abbreviations:
        pretty_words = [
            word.replace("TDC", "Trojan Detection Challenge") for word in pretty_words
        ]

    if verbose:
        print(label)
        print(pretty_words)

    return " ".join(pretty_words)


def split_title(title, max_length, verbose=False):
    """
    Splits a title into multiple lines if it exceeds a certain length. The title is split
    into as many parts as necessary, each not exceeding the maximum length. The split is made at
    the nearest space or underscore before the midpoint of each segment.

    Args:
        title (str): The title to be split.
        max_length (int): The maximum length of each line.

    Returns:
        str: The title, possibly split into multiple lines.
    """
    if len(title) <= max_length:
        return title

    # Split the title into words
    words = title.split()

    lines = []
    current_line = ""

    for word in words:
        # If adding a new word doesn't exceed the max length, add it to the line
        if len(current_line) + len(word) <= max_length:
            current_line += word + " "
        else:
            # If adding a new word exceeds the max length, start a new line
            lines.append(current_line)
            current_line = word + " "

    # Add the last line if it's non-empty
    if current_line:
        lines.append(current_line)

    if verbose:
        print(title)
        print(lines)

    # Join the lines with newlines
    return "\n".join(lines)


def yaxis_formatter(x, pos):
    return f"{x:.2g}"


# Custom comparison function
def not_within_tolerance(a, b, rtol=1e-2, atol=0):
    if np.isnan(a) and np.isnan(b):
        return False
    return not np.isclose(a, b, rtol=rtol, atol=atol)


def get_cf_changing_params(df_params_means):
    """
    The function takes a DataFrame containing parameters and their counterfactual values for different programs,
    and returns a DataFrame of only those parameters that, for any program, change between mainline value and
    counterfactual value.

    Args:
        df_params_means: DataFrame containing parameters and their counterfactual values for different programs
        DataFrame with parameters that change between mainline value and counterfactual value for any program

    Returns:
        df_params_means: DataFrame containing parameters and their counterfactual values for different programs
    """

    # Separate mainline and cf rows by filtering based on the parameter names
    mainline_rows = df_params_means.loc[
        ~df_params_means["parameter"].str.endswith("_cf")
    ]
    cf_rows = df_params_means.loc[
        df_params_means["parameter"].str.endswith("_cf")
    ].copy()

    # Remove '_cf' from parameter names in cf_rows
    cf_rows["parameter"] = cf_rows["parameter"].str.replace("_cf", "")

    rows = []
    for _, row in mainline_rows.iterrows():
        mainline_param = row["parameter"]
        cf_row = cf_rows.loc[cf_rows["parameter"] == mainline_param]

        # If there's no corresponding cf row, append the mainline row and continue
        if cf_row.empty:
            rows.append(row)
            continue

        cf_row = cf_row.iloc[0]

        # Check if any value in row and cf_row is not within tolerance
        not_within_tolerance_flag = False
        for column in df_params_means.columns:
            if column == "parameter":
                continue
            if not_within_tolerance(row[column], cf_row[column]):
                not_within_tolerance_flag = True
                break

        if not_within_tolerance_flag:
            rows.append(row)
            cf_row_copy = cf_row.copy()
            cf_row_copy["parameter"] += "_cf"
            rows.append(cf_row_copy)

    # Concatenate the rows in the desired order
    reshaped_filtered_df = pd.DataFrame(rows)

    return reshaped_filtered_df


def format_number(x):
    """
    Function to format a number based on its size.

    Arguments:
        x -- An integer, float, or NaN.

        If the absolute value of 'x' is in the millions or more, it is formatted with commas as thousand separators.
        If the absolute value of 'x' is less than a million but greater than or equal to 100,000, it is rounded to 3 significant figures.
        If the absolute value of 'x' is less than 100,000 but greater than or equal to 10,000, it is rounded to 2 significant figures.
        If the absolute value of 'x' is less than 10,000 but greater than or equal to 100, it is displayed as a whole number rounded to the nearest whole number.
        If the absolute value of 'x' is less than 100, it is displayed as a float rounded to 2 significant figures.
        NaN values are returned as an empty string.
        Non-numeric values are returned as they are.

    Returns:
        A formatted string representation of 'x'.
    """

    if isinstance(x, (int, float)) and not pd.isna(x):
        # Round the number to 3 significant figures
        rounded_number_3sf = float(f"{x:.3g}")
        if abs(x) >= 1e6:  # If number is in millions or more
            return f"{rounded_number_3sf:,.0f}"  # This will include commas as thousand separators
        else:
            # Round the number to 2 significant figures
            rounded_number_2sf = float(f"{x:.2g}")
            if abs(x) >= 1e5:
                return f"{rounded_number_3sf:,.0f}"
            elif abs(x) >= 1e4:
                return f"{rounded_number_2sf:,.0f}"
            elif abs(x) >= 1e2:
                return f"{rounded_number_2sf:.0f}"
            else:
                return f"{rounded_number_2sf:.2g}"
    elif pd.isna(x):
        return ""
    else:
        return x


def formatted_markdown_table(
    df,
    param_names,
    included_param_names,
    formatting_function,
    descriptions=None,
    bold_rows=[],
    headers="keys",
    tablefmt="pipe",
    showindex=False,
):
    """
    Function to format a DataFrame as a markdown table.

    Args:
        df: DataFrame to be formatted
        param_names: List of all parameter names
        included_param_names: List of parameter names to be included in the table
        formatting_function: Function to be applied to all elements in the DataFrame
        descriptions: Dictionary of parameter descriptions
        bold_rows: List of parameter names to be bolded
        headers: Headers to be used for the table
        tablefmt: Format of the table
        showindex: Whether to show the index of the DataFrame

    Returns:
        A formatted string representation of the DataFrame.
    """
    # Find the intersection of param_names and included_param_names
    final_param_names = list(set(param_names) & set(included_param_names))

    df = df.set_index("parameter")
    df = df.reindex(final_param_names)
    df = df.reset_index()

    # Apply the formatting function to all elements in the DataFrame
    df_formatted = df.applymap(formatting_function)

    # Prettify column names
    df_formatted.columns = [
        prettify_label(col, capitalize_each_word=True) for col in df_formatted.columns
    ]

    # Add descriptions as a second column if provided
    if descriptions is not None:
        df_formatted.insert(
            1, "Description", df_formatted["Parameter"].map(descriptions).fillna("")
        )

    # Order rows by param_names
    df_formatted["Order"] = df_formatted["Parameter"].apply(
        lambda x: param_names.index(x)
    )
    df_formatted = df_formatted.sort_values("Order").drop("Order", axis=1)

    # Prettify parameter names
    df_formatted["Parameter"] = df_formatted["Parameter"].apply(
        lambda x: prettify_label(x, capitalize_each_word=True)
    )

    # Replace df with your DataFrame
    markdown_table = tabulate(
        df_formatted, headers=headers, tablefmt=tablefmt, showindex=showindex
    )

    markdown_lines = markdown_table.splitlines()

    for i, line in enumerate(markdown_lines):
        for bold_row in bold_rows:
            if bold_row in line:
                bold_line = "|".join(
                    [
                        f"**{cell.strip()}**" if cell.strip() else ""
                        for cell in line.split("|")
                    ]
                )
                markdown_lines[i] = bold_line
                break

    bold_markdown_table = "\n".join(markdown_lines)

    return print(bold_markdown_table)


def formatted_markdown_table_research_avenues(
    df,
    index_col,
    param_names,
    formatting_function,
    bold_rows=[],
    headers="keys",
    tablefmt="pipe",
    showindex=False,
):
    """
    Function to format a DataFrame of research avenue relevances as a markdown table.

    Args:
        df: DataFrame to be formatted
        index_col: Name of the column to be used as the index
        param_names: List of all parameter names
        formatting_function: Function to be applied to all elements in the DataFrame
        bold_rows: List of parameter names to be bolded
        headers: Headers to be used for the table
        tablefmt: Format of the table
        showindex: Whether to show the index of the DataFrame

    Returns:
        A formatted string representation of the DataFrame.
    """
    df = df.set_index(index_col)
    df = df.reindex(param_names)
    df = df.reset_index()

    # Apply the formatting function to all elements in the DataFrame
    df_formatted = df.applymap(formatting_function)

    markdown_table = tabulate(
        df_formatted, headers=headers, tablefmt=tablefmt, showindex=showindex
    )

    markdown_lines = markdown_table.splitlines()

    for i, line in enumerate(markdown_lines):
        for bold_row in bold_rows:
            if bold_row in line:
                bold_line = "|".join(
                    [
                        f"**{cell.strip()}**" if cell.strip() else ""
                        for cell in line.split("|")
                    ]
                )
                markdown_lines[i] = bold_line
                break

    bold_markdown_table = "\n".join(markdown_lines)

    return print(bold_markdown_table)


"""
Cost-effectiveness table
"""


def compute_parameter_means(df_params):
    """
    Computes parameter means in preparation for cost-effectiveness table.

    Args:
        df_params (pd.DataFrame): The dataframe containing the parameter values.

    Returns:
        df: The dataframe containing the parameter means.
    """

    # Compute parameter means
    means = {}
    for program, df in df_params.items():
        means[program] = df.mean()
    df_params_means = pd.DataFrame(means)
    df_params_means.reset_index(inplace=True)
    df_params_means.rename(columns={"index": "parameter"}, inplace=True)

    return df_params_means


def calculate_area(
    df, end, researcher_type, participant_type="contender", productivity=None
):
    """
    Calculate the area under the curve of the product of three series in a dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the time series.
        end (int): The end time point.
        researcher_type (str): The type of researcher.
        participant_type (str): The type of participant.
        productivity (pd.Series): The productivity series.

    Returns:
        float: The estimated area under the curve of the product of the three series.
    """

    # Filter dataframe to only include time points between 0 and end
    df_filtered = df[(df["time_point"] >= 0) & (df["time_point"] <= end)]

    # If productivity specified, use this instead of df
    if productivity is None:
        productivity = df_filtered[
            "productivity_over_t_" + participant_type + "_" + researcher_type
        ]
    else:
        productivity = productivity

    # Multiply the three specified columns together to produce new data points
    product_series = (
        productivity
        * df_filtered[
            "p_staying_in_ai_over_t_" + participant_type + "_" + researcher_type
        ]
        * df_filtered[
            "research_time_discount_over_t_" + participant_type + "_" + researcher_type
        ]
    )

    # Estimate the area under the curve using the trapezoidal rule
    area = np.trapz(product_series, df_filtered["time_point"])

    return area


def add_program(df, program_name, cost, benefit):
    """
    Add a program to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add the program to.
        program_name (str): The name of the program.
        cost (float): The cost of the program.
        benefit (float): The benefit of the program.

    Returns:
        pd.DataFrame: The updated DataFrame.
    """
    program_data = {
        "parameter": ["target_budget", "qarys", "qarys_cf"],
        program_name: [cost, benefit, 0],
    }
    df_program = pd.DataFrame(program_data)

    # Merge the new data in by parameter
    df = df.merge(df_program, on="parameter", how="outer")

    return df


def formatted_markdown_table_cost_effectiveness(
    df_params_means,
    param_names,
    formatting_function,
    descriptions=None,
    bold_rows=[],
    extra_programs=None,
):
    """
    Generate a formatted markdown table of cost-effectiveness for the specified programs.

    Params:
        df_params_means: DataFrame containing the mean parameter values including budgets
        param_names: List of parameter names to include in the table
        formatting_function: Function used to format the numbers in the table
        descriptions: Optional, dictionary of program descriptions
        bold_rows: Optional, list of program names for which the rows should be bolded

    Return:
        Formatted markdown table of cost-effectiveness
    """
    if extra_programs is not None:
        for program_name, program_values in extra_programs.items():
            df_params_means = add_program(
                df_params_means, program_name, *program_values
            )

    M = 1e6
    df_filtered = df_params_means[df_params_means["parameter"].isin(param_names)]

    df_filtered.set_index("parameter", inplace=True)
    df_filtered = df_filtered.T

    df_filtered["Benefit (counterfactual expected QARYs)"] = (
        df_filtered["qarys"] - df_filtered["qarys_cf"]
    )
    df_filtered["Cost-effectiveness (QARYs per $1M)"] = df_filtered[
        "Benefit (counterfactual expected QARYs)"
    ] / (df_filtered["target_budget"] / M)
    df_result = df_filtered[
        [
            "target_budget",
            "Benefit (counterfactual expected QARYs)",
            "Cost-effectiveness (QARYs per $1M)",
        ]
    ]
    df_formatted = df_result.applymap(formatting_function)
    df_formatted.columns = [
        "Cost (USD)",
        "Benefit (counterfactual expected QARYs)",
        "Cost-effectiveness (QARYs per $1M)",
    ]
    df_formatted.index.name = "Program"
    df_formatted.index = df_formatted.index.map(
        lambda label: prettify_label(
            label, capitalize_each_word=True, expand_abbreviations=True
        )
    )
    df_formatted.reset_index(inplace=True)

    markdown_table = tabulate(
        df_formatted, headers="keys", tablefmt="pipe", showindex=False
    )

    markdown_lines = markdown_table.splitlines()

    # Calculate the extra spacing needed for non-bold cells to match the bold cells
    extra_spacing = len("**")  # The number of characters added by the bold formatting

    for i, line in enumerate(markdown_lines):
        line_cells = line.split("|")
        for j, cell in enumerate(line_cells):
            # If the cell should be bolded, add the bold formatting
            if any(bold_row in cell for bold_row in bold_rows):
                line_cells[j] = f"**{cell.strip()}**"
            # If the cell should not be bolded, add the extra spacing
            else:
                line_cells[j] = f"{cell.strip():<{len(cell.strip()) + extra_spacing}}"
        # Join the cells back together to form the line
        markdown_lines[i] = "| ".join(line_cells)

    bold_markdown_table = "\n".join(markdown_lines)

    return print(bold_markdown_table)


def formatted_markdown_table_cost_effectiveness_across_stages(
    dfs_params_means,
    param_names,
    formatting_function,
    build_up_stages,
    descriptions=None,
    bold_rows=[],
    extra_programs=None,
):
    """
    Generate a formatted markdown table of cost-effectiveness for the specified programs.
    This function combines multiple dataframes corresponding to different build-up stages.

    Args:
        dfs_params_means (list of pd.DataFrame): List of DataFrames containing the mean parameter values including budgets. Each DataFrame should have a 'parameter' column and a 'target_budget' column.
        param_names (list of str): List of parameter names to include in the table.
        formatting_function (function): Function used to format the numbers in the table.
        build_up_stages (list of str): List of labels for the build-up stages corresponding to the dataframes.
        descriptions (dict, optional): Dictionary of program descriptions. Default is None.
        bold_rows (list of str, optional): List of program names for which the rows should be bolded. Default is [].
        extra_programs (dict, optional): Dictionary of extra programs to add. Each key should be a program name, and each value should be a tuple of program values. Default is None.

    Returns:
        None. This function prints a markdown table of cost-effectiveness.
    """

    M = 1e6

    if extra_programs is not None:
        for program_name in extra_programs.keys():
            program_data = extra_programs[program_name]
            df_extra_programs = pd.DataFrame(
                {
                    "parameter": [program_name],
                    "Cost (USD)": [program_data["cost"]],
                    "Benefit (counterfactual expected QARYs)": [
                        program_data["benefit"]
                    ],
                    "Cost-effectiveness (QARYs per $1M)": [
                        program_data["benefit"] / (program_data["cost"] / M)
                    ],
                },
            )
            df_extra_programs.set_index("parameter", inplace=True)
            df_extra_programs.columns = [
                "Cost (USD)",
                "Benefit (counterfactual expected QARYs)",
                "Cost-effectiveness (QARYs per $1M)",
            ]
        dfs_filtered = [df_extra_programs]
    else:
        dfs_filtered = []

    for df_params_means in dfs_params_means:
        df_filtered = df_params_means[df_params_means["parameter"].isin(param_names)]
        df_filtered.set_index("parameter", inplace=True)
        df_filtered = df_filtered.T

        df_filtered["Benefit (counterfactual expected QARYs)"] = (
            df_filtered["qarys"] - df_filtered["qarys_cf"]
        )
        df_filtered["Cost-effectiveness (QARYs per $1M)"] = df_filtered[
            "Benefit (counterfactual expected QARYs)"
        ] / (df_filtered["target_budget"] / M)
        df_result = df_filtered[
            [
                "target_budget",
                "Benefit (counterfactual expected QARYs)",
                "Cost-effectiveness (QARYs per $1M)",
            ]
        ]
        df_result.columns = [
            "Cost (USD)",
            "Benefit (counterfactual expected QARYs)",
            "Cost-effectiveness (QARYs per $1M)",
        ]
        dfs_filtered.append(df_result)

    # Combine the filtered dataframes with an extra column for the build-up stage
    df_combined = (
        pd.concat(
            [
                df.assign(Build_up_stage=stage)
                for df, stage in zip(dfs_filtered, build_up_stages)
            ],
            axis=0,
        )
        .reset_index()
        .rename(columns={"index": "Program"})
    )

    # Sort by program and build-up stage
    df_combined["Build-up stage"] = pd.Categorical(
        df_combined["Build_up_stage"], categories=build_up_stages, ordered=True
    )
    df_combined.sort_values(["Program", "Build-up stage"], inplace=True)

    # Apply formatting and prettify labels
    df_formatted = df_combined.applymap(formatting_function)
    df_formatted["Program"] = df_formatted["Program"].map(
        lambda label: prettify_label(
            label, capitalize_each_word=True, expand_abbreviations=True
        )
    )

    # Reorder the columns
    df_formatted = df_formatted[
        [
            "Program",
            "Build-up stage",
            "Cost (USD)",
            "Benefit (counterfactual expected QARYs)",
            "Cost-effectiveness (QARYs per $1M)",
        ]
    ]

    markdown_table = tabulate(
        df_formatted, headers="keys", tablefmt="pipe", showindex=False
    )

    markdown_lines = markdown_table.splitlines()

    for i, line in enumerate(markdown_lines):
        line_cells = line.split("|")
        # Check if any cell contains a bold_row
        if any(bold_row in line for bold_row in bold_rows):
            # If so, bold all non-empty cells in the line
            line_cells = [
                f"**{cell.strip()}**" if cell.strip() else cell for cell in line_cells
            ]
        else:
            # If not, remove leading and trailing whitespace from each cell
            line_cells = [cell.strip() for cell in line_cells]
        # Join the cells back together to form the line
        markdown_lines[i] = "|".join(line_cells)

    bold_markdown_table = "\n".join(markdown_lines)

    return print(bold_markdown_table)


def formatted_markdown_table_qarys_per_participant(df_params, formatting_function):
    """
    Reshapes and merges a dictionary of DataFrames into a single DataFrame, more efficiently.

    Args:
        df_params (dict of pd.DataFrame): The dictionary of DataFrames to be reshaped and merged. Keys are assumed to be program names.
        formatting_function (function): The function used to format the numbers.
        prettify_label (function): The function used to prettify labels.

    Returns:
        str: A formatted markdown table.
    """
    data = []

    # Iterate over dictionary items
    for program, df in df_params.items():
        # Find columns that match 'qarys_per_participantType_researcherType' and 'qarys_per_participantType_researcherType_cf'
        for column in df.columns:
            if "qarys_per_" in column:
                participant_type, researcher_type = column.split("_")[2:4]

                # Exclude rows where the participant type is either "during" or "after"
                if participant_type in ["during", "after"]:
                    continue

                qarys_mean = df[column].mean()

                # Determine QARYs type
                if "_cf" in column:
                    qarys_type = "QARYs without program"
                else:
                    qarys_type = "QARYs with program"

                # Add the data to the list
                data.append(
                    [program, participant_type, researcher_type, qarys_type, qarys_mean]
                )

    # Convert the list into a DataFrame
    df_final = pd.DataFrame(
        data,
        columns=[
            "Program",
            "Participant type",
            "Researcher type",
            "QARYs type",
            "Mean QARYs",
        ],
    )

    # Pivot DataFrame to get 'QARYs with program' and 'Counterfactual QARYs' as separate columns
    df_pivot = df_final.pivot_table(
        index=["Program", "Participant type", "Researcher type"],
        columns="QARYs type",
        values="Mean QARYs",
    ).reset_index()

    # Compute 'QARYs without program' as the mean of 'Counterfactual QARYs' minus 'QARYs with program'
    df_pivot["Counterfactual QARYs"] = (
        df_pivot["QARYs with program"] - df_pivot["QARYs without program"]
    )

    # Define order of 'Researcher type'
    researcher_order = ["scientist", "professor", "engineer", "phd"]

    # Set 'Researcher type' as ordered category
    df_pivot["Researcher type"] = pd.Categorical(
        df_pivot["Researcher type"], categories=researcher_order, ordered=True
    )

    # Sort DataFrame by 'Program', 'Participant type', and 'Researcher type'
    df_pivot.sort_values(
        ["Program", "Participant type", "Researcher type"], inplace=True
    )

    # Apply the prettify function to the relevant columns
    df_pivot["Program"] = df_pivot["Program"].apply(prettify_label)
    df_pivot["Participant type"] = df_pivot["Participant type"].apply(prettify_label)
    df_pivot["Researcher type"] = df_pivot["Researcher type"].apply(prettify_label)

    # Apply the formatting function to the numerical columns
    df_pivot["QARYs with program"] = df_pivot["QARYs with program"].apply(
        formatting_function
    )
    df_pivot["QARYs without program"] = df_pivot["QARYs without program"].apply(
        formatting_function
    )
    df_pivot["Counterfactual QARYs"] = df_pivot["Counterfactual QARYs"].apply(
        formatting_function
    )

    # Convert the DataFrame into a markdown table using tabulate
    markdown_table = tabulate(
        df_pivot, headers="keys", tablefmt="pipe", showindex=False
    )

    return print(markdown_table)
