from qubovert import boolean_var, PCBO


def convert_clause_to_penalty(clause: list[tuple[boolean_var, bool]]) -> list[tuple[PCBO, bool]]:
    """
        Coverts a given clause of a SAT instance in conjunctive form to the corresponding penalty
        The clause is encoded as a list of tuples containing a boolean_var
        and False, if the variable is negated or True otherwise

        :param clause: the clause to convert
    """
    penalty = PCBO()
    if len(clause) == 1:
        if clause[0][1]:
            penalty += (1 - clause[0][0])
        else:
            penalty += (clause[0][0])
    elif len(clause) == 2:
        if clause[0][1] and clause[1][1]:
            penalty += (1 - clause[0][0] - clause[1][0] + clause[0][0] * clause[1][0])
        elif clause[0][1] and not clause[1][1]:
            penalty += (clause[1][0] - clause[0][0] * clause[1][0])
        elif not clause[0][1] and clause[1][1]:
            penalty += (clause[0][0] - clause[1][0] * clause[0][0])
        else:
            penalty += (clause[0][0] * clause[1][0])
    else:
        first_two_vars_clause = [clause[0], clause[1]]
        clause_without_first_two_vars = clause
        clause_without_first_two_vars.pop(0)
        clause_without_first_two_vars.pop(0)

        new_clause = []
        new_clause.extend(convert_clause_to_penalty(first_two_vars_clause))
        new_clause.extend(clause_without_first_two_vars)

        penalty += convert_clause_to_penalty(new_clause)[0][0]
    return [(penalty, False)]


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
        penalty += clause_penalty[0][0]
    return penalty
