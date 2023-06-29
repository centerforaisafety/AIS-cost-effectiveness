# Parameters

This directory contains the parameters used in this project.

## Definitions

### Parameter instance

By a parameter 'instance,' we mean a collection of parameters that describe the world as seen from the model. 

A parameter instance (incomplete, for purposes of this example) might look like:

```
params_mainline = {
    "target_budget": 100 * K,
    "p_professor_given_phd": sq.beta(0.1, (1 - 0.1)),
    "scientist_equivalent_professor": 10,
}
```

There will be default parameter instances that describe worlds in which the program is implemented (`mainline`), matched to other instances that describe worlds without the program (`mainline_cf`). There will be parameter instances that describe different stages at which the model is 'built up' (see below), and instances that describe robustness checks (e.g. `larger_labor_costs`). Finally, instances could correspond to different possible program implementations -- one might want to explore a `cheaper_workshop,` or compare `mainline_cais` with different perspectives like `mainline_alice` and `mainline_bob`.

### Parameter set

By a parameter 'set,' we mean a collection of parameter 'instances.'

A parameter set might look like:

```
params = {
    "mainline": params_mainline,
    "mainline_cf": params_mainline_cf,
    "larger_labor_costs": params_larger_labor_costs,
    "larger_labor_costs_cf": params_larger_labor_costs_cf,
}
```

where `_cf` refers to the counterfactual in which the program is not implemented. This `params` object appears towards the end of parameter scripts. It is a two-layer dictionary, where the inner dictionaries are parameter instances, and the outer dictionary contains the various inner dictionaries.

## Role that these scripts play in the project

Each script contains sets of parameter instances for a different program. These parameters will be fed in to [`models`](../models/) using [`sampling`](../utilities/sampling/) functions called by higher-level [`scripts`](../scripts/).

## Structure of the scripts

Each script is structured in a similar pattern.

### Imports

Imports needed to specify or modify parameters.

### Specify parameters, building up from simple assumptions

In this section, we manually specify parameters for the program in question.

First, we specify any variables that would be easier to read if calculated outside of parameter dictionaries. For example, in [`mlss.py`](mlss.py),

```python
variable_cost_per_student = (10 * K + 7 * 4.5 * K) / 7
```

Next, we "build up" the parameter dictionaries in a layered fashion, following the same pattern as our write-ups. Here's a simplified example:

```python
params_build_cost_and_participants = {
    "target_budget": 330 * K,
    "p_pursue_ais": 0.02,
    "research_relevance_student_undergrad_via_phd": 2,
}

additional_params_build_cost_and_participants_cf = {
    "research_relevance_student_undergrad_via_phd": 1,
}

params_build_cost_and_participants_cf = {
    **params_build_cost_and_participants,
    **additional_params_build_cost_and_participants_cf,
}

additional_params_build_pipeline_and_equivalence = {
    "p_pursue_ais": sq.beta(0.1 * 300, (1 - 0.1) * 300),
}

additional_params_build_pipeline_and_equivalence_cf = {
}

params_build_pipeline_and_equivalence = {
    **params_build_cost_and_participants,
    **additional_params_build_pipeline_and_equivalence,
}

params_build_pipeline_and_equivalence_cf = {
    **params_build_pipeline_and_equivalence,
    **additional_params_build_cost_and_participants_cf,
    **additional_params_build_pipeline_and_equivalence_cf,
}
```

In the above example, the first dictionary specifies variables:

1. that will be unchanged in all parameter instances (`target_budget`), 
2. that counterfactually change between instances with or without the program (`research_relevance_student_undergrad_via_phd`), and
3. that begin as simple assumptions, before being 'built up' (`p_pursue_ais`).

(In the actual scripts, several parameters are in both categories 2 and 3.)

The order in which we modify and append dictionaries ensures correct layering of assumptions. For instance, `params_build_cost_and_participants_cf` is created as the full dictionary of parameter assumptions `params_build_cost_and_participants` but with parameters modified according to `additional_params_build_cost_and_participants_cf`. Likewise, `params_build_pipeline_and_equivalence` layers `additional_params_build_pipeline_and_equivalence` on top of initial parameters `params_build_cost_and_participants` without contaminating parameters in `params_build_pipeline_and_equivalence_cf`.

### Default parameters

To make it clear which parameter instances we treat as our all-considered 'defaults', we specify `params_mainline` and `params_mainline_cf` (default parameters with or without the program taking place) explicitly.

### Dictionary of parameter dictionaries

Above, each parameter dictionary (`params_mainline`, `params_build_cost_and_participants`, `params_build_pipeline_and_equivalence`, etc.) is its own 'instance'. In order to correctly feed parameters into the models, we create a dictionary of these different parameter instances. This 'dictionary of dictionaries' is the `params` object.

### Standard robustness checks

Finally, we augment `params` with standardized robustness checks -- for instance, increasing or decreasing the programs' labor costs by some factor. (The full set of currently-implemented possibilities can be found in the `perform_robustness_checks` function in [`utilities/functions/robustness.py`](../../src/utilities/functions/robustness.py).)

## More information

For parameter definitions and values, see the [Parameter Documentation](https://docs.google.com/spreadsheets/d/1uK4opqsCmC5nW6G3D1X67KZnQdMGGL1YbpVmQN5OKF8/edit#gid=581108234) sheet.

