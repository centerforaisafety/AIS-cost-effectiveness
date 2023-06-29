"""
Purpose: functions for calculating the conditional, unconditional, and component 
    probabilities of becoming a scientist-equivalent
"""


class ScientistEquivalentProbability:
    """
    A class that calculates the probability of becoming a scientist-equivalent
    based on various input probabilities.

    Attributes:
        p_scientist_given_phd (float): probability of becoming a scientist given a PhD
        p_professor_given_phd (float): probability of becoming a professor given a PhD
        p_engineer_given_phd (float): probability of becoming an engineer given a PhD
        scientist_equivalent_professor (float): rate at which we would trade off an hour of labor from professor with an hour of otherwise-similar labor from a scientist
        scientist_equivalent_engineer (float): rate at which we would trade off an hour of labor from an engineer with an hour of otherwise-similar labor from a scientist
        p_phd_given_pursue_ais (float): probability of getting a PhD given pursuing AI safety
        p_pursue_ais_given_want (float): probability of pursuing AI safety given wanting to pursue AI safety
        p_want_to_pursue_ais (float): probability of wanting to pursue AI safety
    """

    def __init__(
        self,
        p_scientist_given_phd,
        p_professor_given_phd,
        p_engineer_given_phd,
        scientist_equivalent_professor,
        scientist_equivalent_engineer,
        p_phd_given_pursue_ais,
        p_pursue_ais,
        p_scientist_given_not_phd=0,
        p_professor_given_not_phd=0,
        p_engineer_given_not_phd=0,
    ):
        self.p_scientist_given_phd = p_scientist_given_phd
        self.p_professor_given_phd = p_professor_given_phd
        self.p_engineer_given_phd = p_engineer_given_phd
        self.scientist_equivalent_professor = scientist_equivalent_professor
        self.scientist_equivalent_engineer = scientist_equivalent_engineer
        self.p_phd_given_pursue_ais = p_phd_given_pursue_ais
        self.p_pursue_ais = p_pursue_ais
        self.p_scientist_given_not_phd = p_scientist_given_not_phd
        self.p_professor_given_not_phd = p_professor_given_not_phd
        self.p_engineer_given_not_phd = p_engineer_given_not_phd

    @property
    def p_scientist_equivalent_given_not_phd(self):
        """
        Calculates the probability of becoming a scientist-equivalent given not PhD.

        Returns:
            float: The probability of becoming a scientist-equivalent given not PhD.
        """
        p_scientist_equivalent_professor_given_not_phd = (
            self.p_professor_given_not_phd * self.scientist_equivalent_professor
        )
        p_scientist_equivalent_engineer_given_not_phd = (
            self.p_engineer_given_not_phd * self.scientist_equivalent_engineer
        )

        p_scientist_equivalent = (
            self.p_scientist_given_not_phd
            + p_scientist_equivalent_professor_given_not_phd
            + p_scientist_equivalent_engineer_given_not_phd
        )

        return p_scientist_equivalent

    @property
    def p_scientist_equivalent_given_phd(self):
        """
        Calculates the probability of becoming a scientist-equivalent given a PhD.

        Returns:
            float: The probability of becoming a scientist-equivalent given a PhD.
        """
        p_scientist_equivalent_professor_given_phd = (
            self.p_professor_given_phd * self.scientist_equivalent_professor
        )
        p_scientist_equivalent_engineer_given_phd = (
            self.p_engineer_given_phd * self.scientist_equivalent_engineer
        )

        p_scientist_equivalent = (
            self.p_scientist_given_phd
            + p_scientist_equivalent_professor_given_phd
            + p_scientist_equivalent_engineer_given_phd
        )

        return p_scientist_equivalent

    @property
    def p_scientist_equivalent_not_via_phd(self):
        """
        Calculates the probability of becoming a scientist-equivalent without starting a PhD (e.g., becoming a researcher at Redwood Research after undergrad).

        Returns:
            float: The probability of becoming a scientist-equivalent without starting a PhD.
        """
        return (
            self.p_scientist_equivalent_given_not_phd
            * (1 - self.p_phd_given_pursue_ais)
            * self.p_pursue_ais
        )

    @property
    def p_phd(self):
        """
        Calculates the probability of starting a PhD.

        Returns:
            float: The probability of starting a PhD.
        """
        return self.p_phd_given_pursue_ais * self.p_pursue_ais

    @property
    def p_scientist_equivalent_via_phd(self):
        """
        Calculates the probability of becoming a scientist-equivalent via starting a PhD.

        Returns:
            float: The probability of becoming a scientist-equivalent via starting a PhD.
        """
        return self.p_scientist_equivalent_given_phd * self.p_phd

    @property
    def p_scientist_equivalent(self):
        """
        Calculates the probability of becoming a scientist-equivalent unconditionally.

        Returns:
            float: The probability of becoming a scientist-equivalent unconditionally.
        """
        return (
            self.p_scientist_equivalent_not_via_phd
            + self.p_scientist_equivalent_via_phd
        )
