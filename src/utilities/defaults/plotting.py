"""
Purpose: defaults for plotting
"""


""" 
Colors: all programs
"""

program_colors_all_highlight_movers = {
    "atlas": "#4292c6",
    "mlss": "#984ea3",
    "student_group": "#e7298a",
    "undergraduate_stipends": "#fc4e2a",
    "tdc": "#1b9e77",
    "neurips_social": "#1b9e77",
    "neurips_workshop": "#1b9e77",
}

program_colors_all_categorical = {
    "atlas": "#de2d26",
    "mlss": "#993404",
    "student_group": "#e6550d",
    "undergraduate_stipends": "#ce1256",
    "tdc": "#0570b0",
    "neurips_social": "#3690c0",
    "neurips_workshop": "#8c6bb1",
}


"""
Colors: professional programs
"""

program_colors_professional = {
    "tdc": "#1b9e77",
    "neurips_social": "#d95f02",
    "neurips_workshop": "#984ea3",
}

program_colors_multiple_professional = {
    "tdc": {"contender": "#1b9e77", "attendee": "#1b9e77"},
    "neurips_social": {"contender": "#d95f02", "attendee": "#d95f02"},
    "neurips_workshop": {"contender": "#984ea3", "attendee": "#225ea8"},
}


"""
Colors: student programs
"""

program_colors_student = {
    "atlas": "#e7298a",
    "mlss": "#1b9e77",
    "student_group": "#984ea3",
    "undergraduate_stipends": "#d95f02",
}

program_colors_multiple_student = {
    "atlas": {"student_undergrad": "#e7298a"},
    "mlss": {"student_undergrad": "#1b9e77"},
    "student_group": {"student_undergrad": "#984ea3", "student_phd": "#225ea8"},
    "undergraduate_stipends": {"student_undergrad": "#d95f02"},
}
