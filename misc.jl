""" Miscellaneous functions.
"""

function ln(x)
    """ Shorthand for the natural log function.
    """
    log1p(x - 1)
end
