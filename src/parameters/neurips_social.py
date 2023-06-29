"""
Parameters for neurips_social
"""

"""
Imports
"""

import sys

sys.path.append("src")

# Common assumptions
import utilities.assumptions.assumptions_baserates as adb

# Modify parameters for different robustness scenarios
import utilities.functions.robustness as fr

# Specifying distribution shape or magnitude of parameters
import squigglepy as sq
from squigglepy.numbers import K, M


"""
Specify parameters, building up from simple assumptions
"""

research_relevance_with_program = (
    (2 * 10 + 1 * 5 + 3 * 3 + 10 * 1)
    + (0.2 * 3 + (2 / 10) * 100)
    + ((400 - 2 - 1 - 3 - 10 - 0.2 - 0.2) * 0.1)
) / 400

research_relevance_without_program = (
    (2 * 10 + 1 * 5 + 3 * 3 + 10 * 1)
    + (0 * 3 + 0 * 100)
    + ((400 - 2 - 1 - 3 - 10 - 0 - 0) * 0.1)
) / 400

params_build_cost_and_participants = {
    # Cost
    "target_budget": 5.25 * K,
    "fixed_hours_labor": sq.to(40, 70),
    "average_wage": 50,
    "split_variable_cost_event": 0.95,
    "split_variable_cost_award": 0,
    "sd_variable_cost_event": 100,
    "sd_variable_cost_award": 0,
    "sd_hours_labor": 10,
    "fixed_cost_other": 0,
    "event_max_capacity_per_1000": 200,
    # Number of people
    "n_attendee_scaling_parameter_gamma": 1,
    "n_attendee_scaling_parameter_slope": 40,
    "n_attendee_scaling_parameter_intercept": 0,
    "fraction_attendee_scientist": sq.beta(0.1 * 300, (1 - 0.1) * 300),
    "fraction_attendee_professor": sq.beta(0.2 * 300, (1 - 0.2) * 300),
    "fraction_attendee_engineer": sq.beta(0.15 * 300, (1 - 0.15) * 300),
    # Pipeline and scientist-equivalence
    "p_scientist_given_phd": 1,
    "p_professor_given_phd": 0,
    "p_engineer_given_phd": 0,
    "scientist_equivalent_professor": 1,
    "scientist_equivalent_engineer": 1,
    "scientist_equivalent_phd": 1,
    # Ability
    "ability_at_first_attendee_scientist": 1,
    "ability_at_pivot_attendee_scientist": 1,
    "ability_pivot_point_attendee_scientist": 10,
    "ability_at_first_attendee_professor": 1,
    "ability_at_pivot_attendee_professor": 1,
    "ability_pivot_point_attendee_professor": 10,
    "ability_at_first_attendee_engineer": 1,
    "ability_at_pivot_attendee_engineer": 1,
    "ability_pivot_point_attendee_engineer": 10,
    "ability_at_first_attendee_phd": 1,
    "ability_at_pivot_attendee_phd": 1,
    "ability_pivot_point_attendee_phd": 10,
    # Hours
    "hours_on_entry_per_attendee_scientist": 0,
    "hours_on_entry_per_attendee_professor": 0,
    "hours_on_entry_per_attendee_engineer": 0,
    "hours_on_entry_per_attendee_phd": 0,
    "hours_scientist_per_year": adb.hours_scientist_per_year,
    # Research avenue relevance
    "research_relevance_attendee_scientist": 0.01,
    "research_relevance_during_program_attendee_scientist": 0.01,
    "research_relevance_attendee_professor": 0.01,
    "research_relevance_during_program_attendee_professor": 0.01,
    "research_relevance_attendee_engineer": 0.01,
    "research_relevance_during_program_attendee_engineer": 0.01,
    "research_relevance_attendee_phd": 0.01,
    "research_relevance_multiplier_after_phd_attendee_phd": 1,
    "research_relevance_during_program_attendee_phd": 0.01,
    # Productivity, staying in AI research, and time discounting
    "research_discount_rate": 0,
    "years_since_phd_scientist": 15 - adb.years_in_phd,
    "years_since_phd_professor": 15 - adb.years_in_phd,
    "years_since_phd_engineer": 10 - adb.years_in_phd,
    "years_since_phd_phd": 3 - adb.years_in_phd,
    "years_in_phd": adb.years_in_phd,
    "slope_productivity_life_cycle": 0,
    "pivot_productivity_life_cycle": adb.pivot_productivity_life_cycle,
    "slope_staying_in_ai": 0,
    "pivot_staying_in_ai": adb.pivot_staying_in_ai,
    "end_staying_in_ai": adb.end_staying_in_ai,
    # Flags
    "participant_contender": False,
    "participant_attendee": True,
}

additional_params_build_cost_and_participants_cf = {
    "research_relevance_attendee_scientist": 0,
    "research_relevance_during_program_attendee_scientist": 0,
    "research_relevance_attendee_professor": 0,
    "research_relevance_during_program_attendee_professor": 0,
    "research_relevance_attendee_engineer": 0,
    "research_relevance_during_program_attendee_engineer": 0,
    "research_relevance_attendee_phd": 0,
    "research_relevance_during_program_attendee_phd": 0,
}

params_build_cost_and_participants_cf = {
    **params_build_cost_and_participants,
    **additional_params_build_cost_and_participants_cf,
}

additional_params_build_pipeline_and_equivalence = {
    "p_scientist_given_phd": adb.p_scientist_given_phd,
    "p_professor_given_phd": adb.p_professor_given_phd,
    "p_engineer_given_phd": adb.p_engineer_given_phd,
    "scientist_equivalent_professor": adb.scientist_equivalent_professor,
    "scientist_equivalent_engineer": adb.scientist_equivalent_engineer,
    "scientist_equivalent_phd": adb.scientist_equivalent_phd,
}

additional_params_build_pipeline_and_equivalence_cf = {}

params_build_pipeline_and_equivalence = {
    **params_build_cost_and_participants,
    **additional_params_build_pipeline_and_equivalence,
}

params_build_pipeline_and_equivalence_cf = {
    **params_build_pipeline_and_equivalence,
    **additional_params_build_cost_and_participants_cf,
    **additional_params_build_pipeline_and_equivalence_cf,
}

additional_params_build_relevance_and_ability = {
    "research_relevance_attendee_scientist": research_relevance_with_program,
    "research_relevance_during_program_attendee_scientist": research_relevance_with_program,
    "research_relevance_attendee_professor": research_relevance_with_program,
    "research_relevance_during_program_attendee_professor": research_relevance_with_program,
    "research_relevance_attendee_engineer": research_relevance_with_program,
    "research_relevance_during_program_attendee_engineer": research_relevance_with_program,
    "research_relevance_attendee_phd": research_relevance_with_program,
    "research_relevance_during_program_attendee_phd": research_relevance_with_program,
}

additional_params_build_relevance_and_ability_cf = {
    "research_relevance_attendee_scientist": research_relevance_without_program,
    "research_relevance_during_program_attendee_scientist": research_relevance_without_program,
    "research_relevance_attendee_professor": research_relevance_without_program,
    "research_relevance_during_program_attendee_professor": research_relevance_without_program,
    "research_relevance_attendee_engineer": research_relevance_without_program,
    "research_relevance_during_program_attendee_engineer": research_relevance_without_program,
    "research_relevance_attendee_phd": research_relevance_without_program,
    "research_relevance_during_program_attendee_phd": research_relevance_without_program,
}

params_build_relevance_and_ability = {
    **params_build_cost_and_participants,
    **additional_params_build_pipeline_and_equivalence,
    **additional_params_build_relevance_and_ability,
}

params_build_relevance_and_ability_cf = {
    **params_build_relevance_and_ability,
    **additional_params_build_cost_and_participants_cf,
    **additional_params_build_pipeline_and_equivalence_cf,
    **additional_params_build_relevance_and_ability_cf,
}

additional_params_build_remaining_time_functions = {
    "research_discount_rate": adb.research_discount_rate,
    "slope_productivity_life_cycle": adb.slope_productivity_life_cycle,
    "slope_staying_in_ai": adb.slope_staying_in_ai,
}

additional_params_build_remaining_time_functions_cf = {}

params_build_remaining_time_functions = {
    **params_build_cost_and_participants,
    **additional_params_build_pipeline_and_equivalence,
    **additional_params_build_relevance_and_ability,
    **additional_params_build_remaining_time_functions,
}

params_build_remaining_time_functions_cf = {
    **params_build_remaining_time_functions,
    **additional_params_build_cost_and_participants_cf,
    **additional_params_build_pipeline_and_equivalence_cf,
    **additional_params_build_relevance_and_ability_cf,
    **additional_params_build_remaining_time_functions_cf,
}


"""
Default parameters
"""

params_mainline = params_build_remaining_time_functions
params_mainline_cf = params_build_remaining_time_functions_cf


"""
Dictionary of parameter dictionaries
"""

params = {
    "mainline": params_mainline,
    "mainline_cf": params_mainline_cf,
    "build_cost_and_participants": params_build_cost_and_participants,
    "build_cost_and_participants_cf": params_build_cost_and_participants_cf,
    "build_pipeline_and_equivalence": params_build_pipeline_and_equivalence,
    "build_pipeline_and_equivalence_cf": params_build_pipeline_and_equivalence_cf,
    "build_relevance_and_ability": params_build_relevance_and_ability,
    "build_relevance_and_ability_cf": params_build_relevance_and_ability_cf,
    "build_remaining_time_functions": params_build_remaining_time_functions,
    "build_remaining_time_functions_cf": params_build_remaining_time_functions_cf,
}


"""
Standard robustness checks
"""

checks = [
    "larger_difference_in_scientist_equivalence",
    "smaller_difference_in_scientist_equivalence",
    "larger_labor_costs",
    "smaller_labor_costs",
    "more_senior_attendee_composition",
    "less_senior_attendee_composition",
]

params = fr.perform_robustness_checks(params, checks)
