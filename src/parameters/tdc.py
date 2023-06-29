"""
Parameters for Trojan Detection Challenge
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

params_build_cost_and_participants = {
    # Cost
    "target_budget": 65 * K,
    "fixed_hours_labor": sq.to(50, 170),
    "average_wage": 70,
    "split_variable_cost_event": 0,
    "split_variable_cost_award": 0.95,
    "sd_variable_cost_event": 0,
    "sd_variable_cost_award": 1,
    "sd_hours_labor": 30,
    "fixed_cost_other": 5 * K,
    "event_max_capacity_per_1000": 50,
    # Number of people
    "n_contender_scaling_parameter_gamma": 0.3,
    "n_contender_scaling_parameter_slope": 0.5,
    "n_contender_scaling_parameter_intercept": 0,
    "n_scientist_per_entry": 0.2 * sq.to(0.2, 0.4),
    "n_professor_per_entry": 0.2 * sq.to(0.2, 0.4),
    "n_engineer_per_entry": 0.8 * sq.to(2.2, 4),
    "n_phd_per_entry": 0.2 * sq.to(1.7, 3),
    # Pipeline and scientist-equivalence
    "p_scientist_given_phd": 1,
    "p_professor_given_phd": 0,
    "p_engineer_given_phd": 0,
    "scientist_equivalent_professor": 1,
    "scientist_equivalent_engineer": 1,
    "scientist_equivalent_phd": 1,
    # Ability
    "ability_at_first_contender_scientist": 1,
    "ability_at_pivot_contender_scientist": 1,
    "ability_pivot_point_contender_scientist": 10,
    "ability_at_first_contender_professor": 1,
    "ability_at_pivot_contender_professor": 1,
    "ability_pivot_point_contender_professor": 10,
    "ability_at_first_contender_engineer": 1,
    "ability_at_pivot_contender_engineer": 1,
    "ability_pivot_point_contender_engineer": 10,
    "ability_at_first_contender_phd": 1,
    "ability_at_pivot_contender_phd": 1,
    "ability_pivot_point_contender_phd": 10,
    # Hours
    "hours_on_entry_per_contender_scientist": (1200 / 70) * 1 * ((1 - 0.05) / 2),
    "hours_on_entry_per_contender_professor": (1200 / 70) * 1 * 0.05,
    "hours_on_entry_per_contender_engineer": (1200 / 70) * 1 * 1,
    "hours_on_entry_per_contender_phd": (1200 / 70) * 1 * ((1 - 0.05) / 2),
    "hours_scientist_per_year": adb.hours_scientist_per_year,
    # Research avenue relevance
    "research_relevance_contender_professor": 0.01,
    "research_relevance_during_program_contender_professor": 0.01,
    "research_relevance_contender_scientist": 0.01,
    "research_relevance_during_program_contender_scientist": 0.01,
    "research_relevance_contender_engineer": 0.01,
    "research_relevance_during_program_contender_engineer": 0.01,
    "research_relevance_contender_phd": 0.01,
    "research_relevance_multiplier_after_phd_contender_phd": 1,
    "research_relevance_during_program_contender_phd": 0.01,
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
    "participant_contender": True,
    "participant_attendee": False,
}

additional_params_build_cost_and_participants_cf = {
    "research_relevance_contender_professor": 0,
    "research_relevance_during_program_contender_professor": 0,
    "research_relevance_contender_scientist": 0,
    "research_relevance_during_program_contender_scientist": 0,
    "research_relevance_contender_engineer": 0,
    "research_relevance_during_program_contender_engineer": 0,
    "research_relevance_contender_phd": 0,
    "research_relevance_during_program_contender_phd": 0,
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
    "research_relevance_contender_professor": 5.15,
    "research_relevance_during_program_contender_professor": 10,
    "research_relevance_contender_scientist": 5.15,
    "research_relevance_during_program_contender_scientist": 10,
    "research_relevance_contender_engineer": 0,
    "research_relevance_during_program_contender_engineer": 10,
    "research_relevance_contender_phd": 5.15,
    "research_relevance_multiplier_after_phd_contender_phd": 1,
    "research_relevance_during_program_contender_phd": 10,
}

additional_params_build_relevance_and_ability_cf = {
    "research_relevance_contender_scientist": 5.05,
    "research_relevance_during_program_contender_scientist": 5.05,
    "research_relevance_contender_professor": 5.05,
    "research_relevance_during_program_contender_professor": 5.05,
    "research_relevance_contender_engineer": 0,
    "research_relevance_during_program_contender_engineer": 0,
    "research_relevance_contender_phd": 5.05,
    "research_relevance_during_program_contender_phd": 5.05,
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
    "more_researchers_per_entry",
    "fewer_researchers_per_entry",
    "more_senior_contender_composition",
    "less_senior_contender_composition",
    "more_hours_on_each_entry",
    "fewer_hours_on_each_entry",
]

params = fr.perform_robustness_checks(params, checks)
