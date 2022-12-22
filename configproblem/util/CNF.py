import warnings
from qubovert.sat import AND, OR, NOT
from qubovert import PCBO

from sympy.logic import simplify_logic
from sympy import And, Or, Not
from sympy import Symbol as SympySymbol

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

    def unique_symbols(self):
        symbol_names = set()
        for sym in self.symbols:
            symbol_names.add(sym.name)
        return symbol_names

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

    def unique_symbols(self):
        symbol_names = set()
        for clause in self.clauses:
            symbol_names = symbol_names.union(clause.unique_symbols())
        return symbol_names

    def toSympy(self):
        symbol2sympy_symbol = {}

        for symbol in self.unique_symbols():
            symbol2sympy_symbol[symbol] = SympySymbol(symbol)
        
        sympy_cnf_clauses = [] 
        for clause in self.clauses:

            sympy_clause_symbols = []
            for symbol in clause.symbols:
                sympy_symbol = symbol2sympy_symbol[symbol.name]

                if symbol.negated:
                    sympy_clause_symbols.append(Not(sympy_symbol))
                else:
                    sympy_clause_symbols.append(sympy_symbol)

            sympy_cnf_clauses.append(Or(*sympy_clause_symbols))
        
        sympy_cnf = And(*sympy_cnf_clauses)
        return sympy_cnf

    def fromSympy(self, sympy_cnf:And):
        cnf = CNF()

        for sympy_cnf_clause in sympy_cnf.args:
            clause = Clause()

            if isinstance(sympy_cnf_clause, Or):
                for sympy_symbol in sympy_cnf_clause.args:
                    if isinstance(sympy_symbol, Not):
                        clause.add_symbol(
                            Symbol(sympy_symbol.args[0].name, 
                                negated=True))
                    elif isinstance(sympy_symbol,SympySymbol):
                        clause.add_symbol(Symbol(sympy_symbol.name))
            elif isinstance(sympy_cnf_clause,SympySymbol):
                clause.add_symbol(Symbol(sympy_cnf_clause.name))
            
            cnf.add_clause(clause)

        return cnf

    def simplify(self):
        """Convertes this CNF into sympy to use it's simplify function and then returns a new simplified CNF"""
        print("Simplifying")

        sympy_cnf = self.toSympy()
        # print(sympy_cnf)

        simplified_sympy_cnf = simplify_logic(sympy_cnf, form="cnf")
        success = len(sympy_cnf.atoms(Or)) > len(simplified_sympy_cnf.atoms(Or))
        print("Success: ", success)
        # print(simplified_sympy_cnf)

        simplified_cnf = self.fromSympy(simplified_sympy_cnf)

        if(success):
            print(simplified_cnf)

        return simplified_cnf

    # def to_qubo(self, debug=False):
    #     # Create model
    #     H = PCBO()

    #     # enforce NOTs, i.e. a symbol cannot be true and false simultaneously
    #     for symbol in self.unique_symbols():
    #         H.add_constraint_eq_NOT(symbol, "-"+symbol)

    #     # add clause constraints
    #     for clause in self.clauses:
    #         clause_strs = [str(s) for s in clause.symbols]
    #         print(clause_strs)
    #         print()
    #         print(*clause_strs)
    #         H.add_constraint_eq_OR(*clause_strs) 
    #     if debug: 
    #         print(H) 
    #         print("number of variables:", H.num_binary_variables)
    #         H_solutions = H.solve_bruteforce(all_solutions=True)
    #         print("number of solutions:", len(H_solutions))

        
    #     # transformation to qubo
    #     Q = H.to_qubo()
    #     if debug:
    #         print("Number of QUBO variables:", Q.num_binary_variables, "\n")
    #         Q_solutions = [H.convert_solution(x) for x in Q.solve_bruteforce(all_solutions=True)]
    #         print("number of solutions:", len(Q_solutions))
    #         print("Q solutions", "do" if Q_solutions == H_solutions else "do not", "match the H solutions")

    #     return Q

    def to_qubo(self, debug=False):
        ors = []

        # add clause constraints
        for clause in self.clauses:
            clause_strs = [str(s) for s in clause.symbols]
            # print(clause_strs)
            # print()
            # print(*clause_strs)
            ors.append(OR(*clause_strs)) 

        P = AND(*ors)
        if debug: 
            print(P) 


        
        # transformation to qubo
        Q = P.to_qubo()
        # if debug:
        #     print("Number of QUBO variables:", Q.num_binary_variables, "\n")
        #     Q_solutions = [H.convert_solution(x) for x in Q.solve_bruteforce(all_solutions=True)]
        #     print("number of solutions:", len(Q_solutions))
        #     print("Q solutions", "do" if Q_solutions == H_solutions else "do not", "match the H solutions")

        return Q
