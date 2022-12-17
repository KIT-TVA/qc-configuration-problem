import warnings

class Symbol:
    """ A boolean variable with a name"""
    def __init__(self, name, negated=False) -> None:
        self.name = name
        self.negated = negated # True indicates a negation is present, i.e. !name

    def __eq__(self, __o: object) -> bool:
        if type(__o) is not Symbol:
            return False
        
        if __o.name != self.name:
            return False

        if __o.negated != self.negated:
            return False
        
        return True

    def __lt__(self, other):
        if type(other) != __class__:
            raise TypeError(f"Cannot compare Symbol to {type(other)}")
        return str(self) < str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return "-" + self.name if self.negated else self.name


class Clause:
    """A set of disjunctive Symbols"""
    def __init__(self) -> None:
        self.symbols = set()

    def add_symbol(self, symbol):
        if type(symbol) is str:
                self.symbols.add(Symbol(symbol))
        elif type(symbol) is Symbol:
            self.symbols.add(symbol)
        else:
            warnings.warn(f"Cannot add symbol '{symbol}' of type {type(symbol)} to clause!")

    def add_symbols(self, symbols):
        try:
            for symbol in symbols:
                self.add_symbol(symbol)
        except TypeError:
            self.add_symbol(symbol)

    def remove_symbol(self, symbol):
        if type(symbol) is not Symbol:
            warnings.warn(f"Cannot remove symbol '{symbol}' of type {type(symbol)} from clause! Only Symbol types can be removed.")
            return
        
        self.symbols.remove(symbol)

    def __eq__(self, __o: object) -> bool:
        if type(__o) is not Clause:
            return False

        if len(self.symbols) != len(__o.symbols):
            return False
        
        ordered_symbols = list(self.symbols).sort()
        other_symbols = list(__o.symbols).sort()

        if ordered_symbols is None and other_symbols is None:
            return True
        elif ordered_symbols is None or other_symbols is None:
            return False

        for i, sym in enumerate(ordered_symbols):
            if sym != other_symbols[i]:
                return False

        return True

    def __lt__(self, other):
        if type(other) != __class__:
            raise TypeError(f"Cannot compare Clause to {type(other)}")
        return str(self) < str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        s = ""
        for sym in sorted(self.symbols):
            s += str(sym) + "|"
        return s[:-1]


class CNF:
    """ A set of conjunctive Clauses"""
    def __init__(self) -> None:
        self.clauses = set()

    def add_clause(self, clause):
        if type(clause) is not Clause:
            warnings.warn(f"Cannot add clause '{clause}' of type {type(clause)} to clause! Only Clause types can be added.")
            return
    
        self.clauses.add(clause)

    def add_clauses(self, clauses):
        try:
            for clause in clauses:
                self.add_clause(clause)
        except TypeError:
            self.add_clause(clauses)

        
    def remove_clause(self, clause):
        if type(clause) is not Clause:
            warnings.warn(f"Cannot remove clause '{clause}' of type {type(clause)} from clause! Only Clause types can be removed.")
            return
        self.clauses.remove(clause)

    def __eq__(self, __o: object) -> bool:
        if type(__o) is not Clause:
            return False

        if len(self.clauses) != len(__o.clauses):
            return False
        
        ordered_clauses = sorted(self.clauses)
        other_clauses = sorted(__o.clauses)

        for i, cla in enumerate(ordered_clauses):
            if cla != other_clauses[i]:
                return False

        return True

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        s = ""
        for c in sorted(self.clauses):
            s += f"({c})*"
        return s[:-1]