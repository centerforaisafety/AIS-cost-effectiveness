"""
Parameters for neurips_workshop
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

research_relevance_contender_during_program = (
    (
        (1 + 2 + 1 + 1 + 1 + 0.5 + 0.1)
        + (10 * (0.2 * 107) + 10 * (0.1 * 107) + 1 * (0.3 * 107) + 100 * ((1 / 30)))
    )
    + (100 * 0 + 100 * 0 + 10 * 0)
    + (0.1 * (107 - 7 - 0.2 * 107 - 0.1 * 107 - 0.3 * 107 - (1 / 30) - 0 - 0 - 0))
) / 107

research_relevance_contender_with_program = (
    (
        (1 + 2 + 1 + 1 + 1 + 0.5 + 0.1) * (0.7)
        + (0.1 * 7) * (1 - 0.7)
        + (10 * (0.2 * 107) + 10 * (0.1 * 107) + 1 * (0.3 * 107) + 100 * ((1 / 30)))
    )
    + (100 * ((1 / 40)) + 100 * ((1 / 40)) + 10 * ((1 / 4)))
    + (
        0.1
        * (
            107
            - 7
            - 0.2 * 107
            - 0.1 * 107
            - 0.3 * 107
            - (1 / 30)
            - (1 / 40)
            - (1 / 40)
            - (1 / 4)
        )
    )
) / 107

research_relevance_contender_without_program = (
    (
        (1 + 2 + 1 + 1 + 1 + 0.5 + 0.1) * (0.7)
        + (0.1 * 7) * (1 - 0.7)
        + (10 * (0.2 * 107) + 10 * (0.1 * 107) + 1 * (0.3 * 107) + 100 * ((1 / 30)))
    )
    + (100 * 0 + 100 * 0 + 10 * 0)
    + (0.1 * (107 - 7 - 0.2 * 107 - 0.1 * 107 - 0.3 * 107 - (1 / 30) - 0 - 0 - 0))
) / 107

research_relevance_attendee_with_program = (
    (10 * (0.2 * 400) + 10 * (0.1 * 400) + 1 * (0.3 * 400) + 100 * (1 / 30))
    + (100 * (1 / 10) + 100 * (1 / 10) + 10 * 1)
    + 0.1
    * (400 - 0.2 * 400 - 0.1 * 400 - 0.3 * 400 - (1 / 30) - (1 / 10) - (1 / 10) - 1)
) / 400

research_relevance_attendee_without_program = (
    (10 * (0.2 * 400) + 10 * (0.1 * 400) + 1 * (0.3 * 400) + 100 * (1 / 30))
    + (100 * 0 + 100 * 0 + 10 * 0)
    + 0.1 * (400 - 0.2 * 400 - 0.1 * 400 - 0.3 * 400 - (1 / 30) - 0 - 0 - 0)
) / 400


params_build_cost_and_participants = {
    # Cost
    "target_budget": 110 * K,
    "fixed_hours_labor": sq.to(80, 130),
    "average_wage": 60,
    "split_variable_cost_event": 0.005,
    "split_variable_cost_award": 0.96,
    "sd_variable_cost_event": 100,
    "sd_variable_cost_award": 1,
    "sd_hours_labor": 30,
    "fixed_cost_other": 0,
    "event_max_capacity_per_1000": 50 * M,
    # Number of participants
    "n_contender_scaling_parameter_gamma": 0.45,
    "n_contender_scaling_parameter_slope": 0.5,
    "n_contender_scaling_parameter_intercept": 0,
    "n_attendee_scaling_parameter_gamma": 1,
    "n_attendee_scaling_parameter_slope": 65,
    "n_attendee_scaling_parameter_intercept": 0,
    "n_scientist_per_entry": 0.1 * sq.to(0.6, 1.5),
    "n_professor_per_entry": 0.9 * sq.to(0.6, 1.5),
    "n_engineer_per_entry": 0.1 * sq.to(2, 4.5),
    "n_phd_per_entry": 0.9 * sq.to(2, 4.5),
    "fraction_attendee_scientist": sq.beta(0.01 * 300, (1 - 0.01) * 300),
    "fraction_attendee_professor": sq.beta(
        0.15 * 300, (1 - 0.15) * 300
    ),  # easier to implement than 1 / sq.to(4, 11)
    "fraction_attendee_engineer": sq.beta(0.01 * 300, (1 - 0.01) * 300),
    "fraction_attendee_also_contender": 0.1,
    # Pipeline and scientist-equivalents
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
    "hours_on_entry_per_contender_scientist": 60,
    "hours_on_entry_per_contender_professor": 20,
    "hours_on_entry_per_contender_engineer": 47,
    "hours_on_entry_per_contender_phd": 60,
    "hours_scientist_per_year": adb.hours_scientist_per_year,
    # Research avenue relevance
    "research_relevance_contender_scientist": 0.01,
    "research_relevance_during_program_contender_scientist": 0.01,
    "research_relevance_contender_professor": 0.01,
    "research_relevance_during_program_contender_professor": 0.01,
    "research_relevance_contender_engineer": 0.01,
    "research_relevance_during_program_contender_engineer": 0.01,
    "research_relevance_contender_phd": 0.01,
    "research_relevance_multiplier_after_phd_contender_phd": 1,
    "research_relevance_during_program_contender_phd": 0.01,
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
    "research_discount_rate": 0,
    # Flags for which calculations to perform
    "participant_contender": True,
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
    "research_relevance_contender_scientist": 0,
    "research_relevance_during_program_contender_scientist": 0,
    "research_relevance_contender_professor": 0,
    "research_relevance_during_program_contender_professor": 0,
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
    "research_relevance_contender_scientist": research_relevance_contender_with_program,
    "research_relevance_during_program_contender_scientist": research_relevance_contender_during_program,
    "research_relevance_contender_professor": research_relevance_contender_with_program,
    "research_relevance_during_program_contender_professor": research_relevance_contender_during_program,
    "research_relevance_contender_engineer": research_relevance_contender_with_program,
    "research_relevance_during_program_contender_engineer": research_relevance_contender_during_program,
    "research_relevance_contender_phd": research_relevance_contender_with_program,
    "research_relevance_during_program_contender_phd": research_relevance_contender_during_program,
    "research_relevance_attendee_scientist": research_relevance_attendee_with_program,
    "research_relevance_during_program_attendee_scientist": research_relevance_attendee_with_program,
    "research_relevance_attendee_professor": research_relevance_attendee_with_program,
    "research_relevance_during_program_attendee_professor": research_relevance_attendee_with_program,
    "research_relevance_attendee_engineer": research_relevance_attendee_with_program,
    "research_relevance_during_program_attendee_engineer": research_relevance_attendee_with_program,
    "research_relevance_attendee_phd": research_relevance_attendee_with_program,
    "research_relevance_during_program_attendee_phd": research_relevance_attendee_with_program,
}

additional_params_build_relevance_and_ability_cf = {
    "research_relevance_contender_scientist": research_relevance_contender_without_program,
    "research_relevance_during_program_contender_scientist": research_relevance_contender_without_program,
    "research_relevance_contender_professor": research_relevance_contender_without_program,
    "research_relevance_during_program_contender_professor": research_relevance_contender_without_program,
    "research_relevance_contender_engineer": research_relevance_contender_without_program,
    "research_relevance_during_program_contender_engineer": research_relevance_contender_without_program,
    "research_relevance_contender_phd": research_relevance_contender_without_program,
    "research_relevance_during_program_contender_phd": research_relevance_contender_without_program,
    "research_relevance_attendee_scientist": research_relevance_attendee_without_program,
    "research_relevance_during_program_attendee_scientist": research_relevance_attendee_without_program,
    "research_relevance_attendee_professor": research_relevance_attendee_without_program,
    "research_relevance_during_program_attendee_professor": research_relevance_attendee_without_program,
    "research_relevance_attendee_engineer": research_relevance_attendee_without_program,
    "research_relevance_during_program_attendee_engineer": research_relevance_attendee_without_program,
    "research_relevance_attendee_phd": research_relevance_attendee_without_program,
    "research_relevance_during_program_attendee_phd": research_relevance_attendee_without_program,
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
    "more_senior_attendee_composition",
    "less_senior_attendee_composition",
    "more_hours_on_each_entry",
    "fewer_hours_on_each_entry",
]

params = fr.perform_robustness_checks(params, checks)
