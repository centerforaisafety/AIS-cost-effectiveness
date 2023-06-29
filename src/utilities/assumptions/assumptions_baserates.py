"""
Purpose: background assumptions and base-rates used throughout
"""


"""
Imports
"""

import squigglepy as sq
from squigglepy.numbers import K, M


""" 
Assumptions
"""

# Scientist-equivalence
scientist_equivalent_professor = 10
scientist_equivalent_engineer = 0.1
scientist_equivalent_phd = 0.1

# Hours worked
hours_scientist_per_year = 2 * K

# Productivity
slope_productivity_life_cycle = -0.45
pivot_productivity_life_cycle = -3

# Probability of staying in AI
slope_staying_in_ai = -0.15
pivot_staying_in_ai = 20
end_staying_in_ai = 60

# Time discounting
research_discount_rate = 0.2


"""
Base-rates
"""

# The probability of someone completing their PhD given that they pursue AI safety (AIS).
p_phd_given_pursue_ais = sq.beta(0.05 * 300, (1 - 0.05) * 300)

# Job market probabilities
p_scientist_given_phd = sq.beta(0.15 * 1000, (1 - 0.15) * 1000)  # mean 0.15
p_professor_given_phd = sq.beta(0.05 * 1000, (1 - 0.05) * 1000)  # mean 0.05
p_engineer_given_phd = sq.beta(0.15 * 1000, (1 - 0.15) * 1000)  # mean 0.15

# The duration of a PhD in years.
years_in_phd = 6
