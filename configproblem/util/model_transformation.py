from qubovert import boolean_var, PCBO


def convert_clause_to_penalty(clause: list[tuple[boolean_var, bool]]) -> list[tuple[PCBO, bool]]:
    """
        Coverts a given clause of a SAT instance in conjunctive form to the corresponding penalty
        The clause is encoded as a list of tuples containing a boolean_var
        and False, if the variable is negated or True otherwise

        :param clause: the clause to convert
    """
    penalty = PCBO() + 1

    for var in clause:
        if var[1]:
            penalty *= (1 - var[0])
        else:
            penalty *= (var[0])
    return penalty, False


def convert_to_penalty(sat_instance: list[list[tuple[boolean_var, bool]]]) -> PCBO:
    """
        Converts a given SAT instance in conjunctive form to the corresponding penalty
        The SAT instance is encoded as a list of clauses.
        A clause is encoded as a list of tuples containing a boolean_var
        and False if the variable is negated or True otherwise

        :param sat_instance: the SAT instance to convert
    """
    penalty = PCBO()
    for clause in sat_instance:
        clause_penalty = convert_clause_to_penalty(clause)
        penalty += clause_penalty[0]
    return penalty
