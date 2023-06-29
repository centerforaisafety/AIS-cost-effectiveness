# Examples

This directory contains examples of how to use the project. Each script demonstrates a different use case or feature.

In this README, we first give advice for three major use-cases:

1. Probing our published results.
2. Evaluating new programs using our existing models.
3. Evaluating new programs using new models.

We then describe example features, which use-cases might draw upon:

1. Modifying global (constant across programs) parameters.
2. Modifying program-specific parameters.
3. Plotting comparisons between programs.
4. Plotting robustness to individual parameters.
5. Plotting robustness to discrete 'scenarios' (changes to many parameters).
6. Plotting cost-effectiveness at different 'scales' (budget amounts).
7. Printing summary statistics.

If you would like assistance with this repo and/or your own evaluations, contact CAIS at [contact@safe.ai](mailto:contact@safe.ai).

## Major use-cases

### Probe our published results

[**`probe_cais_results.py`**](probe_cais_results.py) provides an example of how to probe the results from our public write-ups. The key steps are:

1. Simulate results for default parameters.
2. Specify various modifications to global (constant across programs) and/or program-specific parameters.
3. Using these modifications, modify the default parameters to obtain new parameter instances. (The [parameters README](../../parameters/README.md) describes what we mean by instances, as well as how and why parameter scripts are structured as they are.)
4. Simulate results using the modified parameter instances.
5. Print a cost-effectiveness table, with results for every program and modification stage (default parameters, global parameter modifications, program-specific parameter modifications, and global and program-specific parameter modifications).
6. Plot these results.

After following this example, you might want to read the other examples in this directory, or to read our [`posts`](../posts/) scripts to understand how we analyze the results in our published posts.

### Evaluate new program using existing models

[**`evaluate_new_program.py`**](evaluate_new_program.py) provides an example of how to evaluate a new program using models in this repository.

In order for the program you want to evaluate to be a good fit for our existing models, it needs to be recognizable as a student program (pay costs, have some cohort of students, change student outcomes counterfactually following the program) or a professional program with an event and/or award component (pay costs, have some cohort of attendees and/or award contenders, change what participants work on counterfactually during and/or after the program). Your program does not obviously need to be related to AI safety research in particular.

In [`evaluate_new_program.py`](evaluate_new_program.py), we evaluate a hypothetical retreat for PhD students. The key steps are:

1. Specify parameter instances with and without the program. (The [parameters README](../../parameters/README.md) describes what we mean by instances, as well as how and why parameter scripts are structured as they are.)
2. Obtain results using these instances.
3. Summarize key parameter values.

The first step largely copies the structure of scripts in the [`parameters`](../../parameters/) folder.

After following this example, you might want to read the other examples in this directory, or to read our [`posts`](../posts/) scripts for examples of how we analyze results.

### Evaluate new program using new models

Read the existing [`models`](../../models/) scripts and the [`utilities`](../../utilities/) provided for them (in particular, lower-level functions called by the models).

You might then want to:

1. Build your own models, copying our existing models and utilities as a starting step.
2. Gather your parameter value beliefs for the programs you want to evaluate.
3. Create a [`parameters`](../../parameters/) script specifying the values of these parameters (and confirming your understanding of parameter definitions using the [Parameter Documentation](https://docs.google.com/spreadsheets/d/1uK4opqsCmC5nW6G3D1X67KZnQdMGGL1YbpVmQN5OKF8/edit#gid=581108234) sheet).
4. Explore results with the help of scripts in this folder, or by following our [`posts`](../posts/) scripts.

## Features

### [`modify_global_parameters`](features/modify_global_parameters.py)

This script shows you how to modify **global** parameters -- parameters with values that are shared across programs.

After importing necessary packages, the key steps are:

1. Specify parameter modifications.
2. List the parameter sets for programs you wish to modify.
3. List the parameter instances you wish to modify for each program. (The [parameters README](../../parameters/README.md) describes what we mean by parameter sets and instances.)
4. Modify parameters in a loop over parameter sets and instances.

```python
# 1. Specify parameter modifications
scientist_equivalent_professor_modified = sq.to(1, 100)

# 2. List parameter sets to modify
parameter_sets = [p_p, p_so, p_w]

# 3. List parameter instances to modify for each program
defaults = ["mainline", "mainline_cf"]

# 4. Loop through parameter sets, modifying the parameter
for parameter_set in parameter_sets:
    for default in defaults:
        params = parameter_set.params[default]
        params[
            "scientist_equivalent_professor"
        ] = scientist_equivalent_professor_modified
        parameter_set.params[
            f'modification_name{"" if default == "mainline" else "_cf"}'
        ] = params
```

### [`modify_program_parameters`](features/modify_global_parameters.py)

This script shows you how to modify **program-specific** parameters -- parameters with values that vary across (and possibly within) programs.

After importing necessary packages, the key steps are:

1. Specify parameter modifications.
2. List the parameter instances you wish to modify for the program you wish to modify. (The [parameters README](../../parameters/README.md) describes what we mean by parameter instances.)
3. Modify parameters in a loop over parameter instances.

```python
# 1. Modify program-specific parameters
students_per_instructor_modified = 10
students_per_instructor_modified_cf = students_per_instructor_modified
p_phd_given_pursue_ais_modified = sq.beta(20, 100)
p_phd_given_pursue_ais_modified_cf = sq.beta(10, 100)

# 2. List parameter instances to modify
defaults = ["mainline", "mainline_cf"]

# 3. Loop through parameter sets, modifying the parameters
for default in defaults:
    params = p_m.params[default]

    if default == "mainline":
        params["students_per_instructor"] = students_per_instructor_modified
        params["p_phd_given_pursue_ais"] = p_phd_given_pursue_ais_modified
    else:
        params["students_per_instructor"] = students_per_instructor_modified_cf
        params["p_phd_given_pursue_ais"] = p_phd_given_pursue_ais_modified_cf

    p_m.params[f'modification_name{"" if default == "mainline" else "_cf"}'] = params
```

### [`plot_program_comparisons`](features/plot_program_comparisons.py)

This script shows you how to explore how to plot comparisons between programs. (The script focuses on the number of participants, ability as a function of number of participants, and QARYs per scientist-equivalent participant over time. For other types of plots, you might want to read the scripts that produce plots for our [`posts`](../posts/).)

After importing necessary packages and specifying objects needed to simulate and plot results, the key steps are:

1. Simulate results.
2. Generate and save the number of participants plot.
3. Generate and save the ability plot.
4. Generate and save the QARYs over time plot.


### [`robustness_to_parameters`](features/robustness_to_parameters.py)

This script shows you how to test the robustness of results to specific parameters.

After importing necessary packages and specifying objects needed to simulate and plot results, the key steps are:

1. Specify the names and values of parameters you want to test the robustness of results with respect to. Make sure to _exclude_ the default value of the parameter. (In the example script, this would be `research_discount_rate = 0.2`.)
2. Simulate and process results for different values of the parameters.
3. Generate the plot.
4. Save the plot.

```python
# 1. Specify the names and values of parameters to change, excluding default
robustness_changes = {"research_discount_rate": [-1, -0.5, -0.2, 0, 0.5]}

# 2. Simulate results for different values of the parameters
df_robustness_research_discount_rate = rob.process_robustness_data(
    programs,
    robustness_changes,
    default_parameters,
    master_functions,
    n_sim=100 * K,
)

# 3. Generate the plot
plot_robustness_research_discount_rate = rob.plot_robustness_multiple(
    df_robustness_research_discount_rate,
    program_colors,
    global_ylim=False,
    global_ylim_bottom=-1 * M,
    title="",
    axis_text_skip=1,
    ylim_buffer=0.5,
)

# 4. Save the plot to a file
plot_robustness_research_discount_rate.set_size_inches((6, 4))
plot_robustness_research_discount_rate.savefig(
    "output/plots/examples/robustness_research_discount_rate.png",
    dpi=300,
    bbox_inches="tight",
)
```

### [`robustness_to_scenarios`](features/robustness_to_scenarios.py)

This script shows you how to test the robustness of results to various (specified) scenarios.

After importing necessary packages and specifying objects needed to simulate and plot results, the key steps are:

1. Specify scenarios to test robustness to. (The full set of currently-implemented possibilities can be found in the `perform_robustness_checks` function in [`utilities/functions/robustness.py`](../../utilities/functions/robustness.py).)
2. Simulate results.
3. Process the results for plotting.
4. Generate the plot.
5. Save the plot.

```python
# 1. Specify scenarios
scenarios = [
    "mainline",
    "larger_difference_in_scientist_equivalence",
    "smaller_difference_in_scientist_equivalence",
    "larger_labor_costs",
    "smaller_labor_costs",
    "larger_fixed_costs",
    "smaller_fixed_costs",
    "better_job_prospects",
    "worse_job_prospects",
]

# 2. Get data
data_robustness = so.get_scenario_data(
    programs=programs,
    default_parameters=default_parameters,
    master_functions=master_functions,
    n_sim=n_sim,
    time_points=time_points,
    scenarios=scenarios,
    create_df_functions=False,
    share_params_cf=False,
)

# 3. Process data for plotting
data_robustness_processed = p_rob.transform_data_robustness_to_df(data_robustness)

# 4. Plot
plot_robustness_scenarios = p_rob.plot_benefit_cost_effectiveness_robustness_charts(
    data_robustness_processed, program_colors
)

# 5. Save
plot_robustness_scenarios.set_size_inches((10, 5))
plot_robustness_scenarios.savefig(
    "output/plots/examples/robustness_scenarios.png",
    dpi=300,
    bbox_inches="tight",
)
```

### [`scalability`](features/scalability.py)

This script shows you how to explore how the cost-effectiveness of and number of participants involved in programs scales with the (target) budget allocated to them.

After importing necessary packages and specifying objects needed to simulate and plot results, the key steps are:

1. Specify possible budget values.
2. Simulate results.
3. Generate and save the cost-effectiveness (and benefit) plot.
4. Generate and save the number of participants plot.

### [`summary_statistics`](features/summary_statistics.py)

This script shows you how to generate summary statistics about programs.

After importing necessary packages and specifying objects needed to simulate results, the key steps are:

1. Simulate results.
2. Generate the inputs needed for `formatted_markdown_table_cost_effectiveness`.
3. Print summary statistics using `formatted_markdown_table_cost_effectiveness`.
