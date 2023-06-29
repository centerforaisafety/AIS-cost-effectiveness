# Models

This directory contains the models used in this project.

## Role that these scripts play in the project

Each script contains a different category of model. Each model is used by multiple programs within a category. (For instance, all student programs work with the [`student_program`](student_program.py) model.)

These models will receive input [`parameters`](../parameters/) using [`sampling`](../utilities/sampling/) functions called by higher-level [`scripts`](../scripts/).

## Structure of the scripts

Each script is structured in a similar pattern.

### Imports

Import lower-level functions called by the models, as well as libraries for vectorization and specifying distributions.

### Cost function

This section contains a function for calculating different cost components of a program. The typical process for calculating costs is as follows:

1. Calculate fixed costs, using pre-specified parameters.
2. Using the supplied "target" budget, calculate the target variable cost (i.e. remaining cost).
3. Calculate the targeted components of variable cost (e.g. cost of students, or events, or awards).
4. Calculate the "actual" (i.e. realized) components of variable cost using pre-specified uncertainty to back out parameters for a gamma distribution.
5. Aggregate actual variable cost components to get actual variable cost.
6. Add back fixed costs to get actual budget.

### Benefit functions

This section contains functions for calculating the benefit of a program.

The first function is `benefit_number_participant`, which calculates the number of participants (and scientist-equivalent participants) given the cost parameters calculated previously.

The second function is `benefit_qarys_per_participant`, which calculates the number of QARYs per (scientist-equivalent) participant of each type. For professional programs, these types might be "attendees" or "contenders" who are currently research "scientists", research "professors", research "engineers", or "PhD" students. For student programs, these types might be "undergraduates" who become researchers "via PhD" or "not via PhD", and current "PhD" students.

The final function, `benefit_mfn`, is an intermediate master function aggregating the earlier two functions.

### Master function

`mfn` is the final and most important function in each script. It samples parameters, calls the other cost and benefit functions discussed above, then merges the counterfactual and non-counterfactual parameters into a single dataframe.

## Objects passed around

The functions above pass around a (mostly) common collection of objects:

1. `params`, the raw parameters supplied to the model. (`params_cf` are analogous parameters referring to the counterfactual in which a program does not take place.) This is a single parameter instance from [`parameters`](../parameters/), selected by a [`sampling`](../utilities/sampling/) function called by higher-level [`scripts`](../scripts/). (The [parameters README](/src/parameters/README.md) describes what we mean by instances.)
2. `params_sampled`, the realizations of possibly-uncertain `params`. (Analogously, `params_sampled_cf`.)
3. `derived_params_sampled`, the realizations of parameters derived within (as opposed to supplied to) the model. (Analogously, `derived_params_sampled_cf`.)
4. `derived_functions`, functions derived within the model. (Analogously, `derived_functions_cf`.)
5. `n_sim`, the number of simulations (i.e. number of realizations of each parameter).
