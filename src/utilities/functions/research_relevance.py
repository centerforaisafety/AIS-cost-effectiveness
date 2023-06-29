"""
Purpose: functions describing research relevance over time
"""


"""
Research avenue relevance
"""


def research_relevance_geometric_over_t(
    itn_baseline,
    itn_endline,
    p_influence_drift,
    years_until_phd=0,
    end=100,
    years_in_program=None,
    research_relevance_during_program=None,
):
    """
    This function computes a researcher's value over time using a geometric model. The value depends on several factors:
    baseline and endline values, influence drift, years until the PhD, and an optional duration in a program with its associated value.

    The function is defined within a given period (from 'years_until_phd' to 'end'). If the time 't' is outside this period, the value is zero.
    If a program duration and a corresponding research relevance during this program are provided, the function returns this research relevance for the time 't' within the program duration.
    Otherwise, it computes the research relevance based on the baseline, endline, influence drift, and time 't'.

    Args:
        itn_baseline (float): The initial value (baseline) for the researcher.
        itn_endline (float): The final value (endline) for the researcher.
        p_influence_drift (float): The influence drift percentage.
        years_until_phd (int, optional): The number of years until the researcher gets a PhD. Default is 0.
        end (int, optional): The end of the period for the function definition. Default is 100.
        years_in_program (int, optional): The duration of a program. Default is None.
        research_relevance_during_program (float, optional): The research relevance during the program. Default is None.

    Returns:
        function: A function 'f' that takes time 't' as an argument and computes the research relevance at this time point.
    """

    def f(t):
        if t < years_until_phd or t > end:
            return 0
        else:
            if (
                years_in_program is not None
                and research_relevance_during_program is not None
            ):
                if years_until_phd <= t < years_in_program:
                    return research_relevance_during_program

            return (
                itn_baseline
                + (itn_endline - itn_baseline) * (1 - p_influence_drift) ** t
            )

    return f


def research_relevance_piecewise_over_t(
    research_relevance,
    research_relevance_multiplier_after_phd,
    years_in_phd=6,
    years_until_phd=0,
    end=100,
    years_in_program=None,
    research_relevance_during_program=None,
):
    """
    This function calculates the piecewise research relevance over time, considering periods during and after a PhD program.
    The value can change during the PhD program, after it, and during an optional program.

    Args:
        research_relevance (float): Base value of research during the PhD program.
        research_relevance_multiplier_after_phd (float): Multiplier for the research relevance after the PhD.
        years_in_phd (int): Duration of the PhD program in years. Default is 6.
        years_until_phd (int): Number of years until the start of the PhD program. Default is 0.
        end (int): The end of the time frame for which the research relevance is calculated. Default is 100.
        years_in_program (int): Optional, number of years in another program which affects the research relevance.
        research_relevance_during_program (float): Optional, value of research during the aforementioned program.

    Returns:
        function: A piecewise function `f(t)` which computes the research relevance at time `t`,
                  taking into account the different periods and their respective multipliers.
    """

    phd_end = years_until_phd + years_in_phd

    # Check if the researcher is not yet finished with their PhD as of time=0
    phd_not_finished = phd_end > 0

    def f(t):
        if t < 0 or t > end:
            return 0
        else:
            if (
                years_in_program is not None
                and research_relevance_during_program is not None
            ):
                if years_until_phd <= t < years_in_program:
                    return research_relevance_during_program

            if t <= phd_end:  # During the PhD
                return research_relevance
            else:  # After the PhD
                if phd_not_finished:
                    return research_relevance * research_relevance_multiplier_after_phd
                else:
                    return research_relevance

    return f
