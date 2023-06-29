"""
Purpose: relevance-to-CAIS of different research avenues
"""


"""
Imports
"""

import pandas as pd
import numpy as np


"""
X-Risk Analysis for AI Research scores

The DataFrame represents a table of different AI-related topics
(the keys in the dictionary) and their respective scores for 
 "importance", "neglectedness", and "tractability".
"""

df = pd.DataFrame(
    {
        "adversarial_robustness": [3, 1, 2],
        "anomaly_detection": [3, 2, 2],
        "interpretable_uncertainty": [2, 1, 2],
        "transparency": [3, 1, 1],
        "trojans": [3, 2, 2],
        "honest_ai": [3, 3, 2],
        "power_aversion": [3, 3, 2],
        "moral_decision_making": [3, 3, 2],
        "value_clarification": [3, 3, 1],
        "ml_for_cyberdefense": [2, 3, 3],
        "ml_for_improving_epistemics": [2, 3, 2],
        "cooperative_ai": [3, 3, 1],
    },
    index=["importance", "neglectedness", "tractability"],
).transpose()


"""
Without compute 
"""

# Compute the ITN score for each row
relevance_score = (
    np.power(10, df["importance"])
    * np.power(10, df["neglectedness"])
    * np.power(10, df["tractability"])
)

# Normalize ITN score relative to adversarial robustness
relevance_normalized = relevance_score / relevance_score["adversarial_robustness"]
df["relevance_normalized"] = relevance_normalized

# Relevance multipliers from each research avenue being pursued with access to industry-scale compute
compute_multipliers = {
    "adversarial_robustness": 3,
    "anomaly_detection": 10,
    "interpretable_uncertainty": 10,
    "transparency": 10,
    "trojans": 10,
    "honest_ai": 1,
    "power_aversion": 1,
    "moral_decision_making": 1,
    "value_clarification": 1,
    "ml_for_cyberdefense": 3,
    "ml_for_improving_epistemics": 3,
    "cooperative_ai": 3,
}

df_expanded = pd.DataFrame(columns=df.columns)

for idx, row in df.iterrows():
    with_compute_row = row.copy()
    without_compute_row = row.copy()
    with_compute_row.name = f"{idx}_with_compute"
    without_compute_row.name = f"{idx}_without_compute"
    multiplier = compute_multipliers[idx]
    with_compute_row["relevance_normalized"] *= multiplier
    df_temp = pd.concat([with_compute_row, without_compute_row], axis=1).T
    df_expanded = pd.concat([df_expanded, df_temp])

impact_research_avenues = {}

for idx, row in df_expanded.iterrows():
    key = f"impact_{idx}"
    value = row["relevance_normalized"]
    impact_research_avenues[key] = value

impact_research_avenues["impact_nothing"] = 0.0
impact_research_avenues["impact_dangerous_capabilities"] = -100.0

impact_df = pd.DataFrame(
    list(impact_research_avenues.items()), columns=["Research Avenue", "Impact"]
)
