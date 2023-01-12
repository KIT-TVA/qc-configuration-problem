import warnings
from qubovert.sat import AND, OR, NOT
from qubovert import PCBO

from sympy.logic.boolalg import to_cnf
from sympy import And, Or, Not
from sympy import Symbol as SympySymbol

from . import dimacs_reader

class Symbol:
    """ A boolean variable with a name"""
    def __init__(self, name, negated=False) -> None:
        self.name = str(name)
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

        if clause in self.clauses:
            warnings.warn(f"Will not add clause {clause} because it is already contained in the CNF!")

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

    def to_sympy(self):
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

    def from_sympy(self, sympy_cnf:And):
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
        sympy_cnf = self.to_sympy()
        simplified_sympy_cnf = to_cnf(sympy_cnf, simplify=True, force=True)
        success = len(sympy_cnf.atoms(Or)) > len(simplified_sympy_cnf.atoms(Or))
        # print("Simplyfing succes: ", success)

        simplified_cnf = self.from_sympy(simplified_sympy_cnf)

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

    def to_problem(self, sympy_constraints=None):
        """Transform this CNF instance to a suitable representation for our Grover Notebooks"""
        problem = []
        symbol_index = {}
        index_counter = 0

        for clause in self.clauses:
            problem_clause = []
            
            for symbol in clause.symbols:
                q_idx = symbol_index.get(symbol.name)
                if q_idx is None:
                    # Add the symbol to the dictionary
                    q_idx = index_counter
                    symbol_index[symbol.name] = index_counter
                    index_counter += 1

                problem_clause.append((q_idx, not symbol.negated))
                
            problem.append(problem_clause)
        
        return problem

    def from_dimacs(self, dimacs_reader: dimacs_reader.DimacsReader):
        cnf = CNF()
        for dimacs_clause in dimacs_reader.clauses:
            clause = Clause()
            for dimacs_symbol in dimacs_clause:
                symbol_id = abs(dimacs_symbol)
                symbol_name = f"dimacs_{symbol_id}"
                
                if dimacs_symbol < 0:
                    # negated symbol
                    sym = Symbol(symbol_name, negated=True)
                    clause.add_symbol(sym)
                elif dimacs_symbol > 0:
                    # regular symbol
                    sym = Symbol(symbol_name)
                    clause.add_symbol(sym)
            
            # add non empty clause to cnf
            if len(clause.symbols) > 0:
                cnf.add_clause(clause)
            else:
                print(f"Hint: Did not add dimacs clause: {dimacs_clause}")

        # Sanity check
        symbols = cnf.unique_symbols()
        nClauses = len(cnf.clauses)
        if len(symbols) != dimacs_reader.nFeatures:
            warnings.warn(f"Feature Mismatch between dimacs input and generated CNF! dimacs {dimacs_reader.nFeatures} / cnf {len(symbols)}")
        if nClauses != dimacs_reader.nClauses:
            warnings.warn(f"Clauses Mismatch between dimacs input and generated CNF! dimacs {dimacs_reader.nClauses} / cnf {nClauses}")

        return cnf

class TestCNF2Problem:
    def test_cnf_to_problem(self):
        f = CNF()
        c1 = Clause()
        c2 = Clause()
        
        c1.add_symbols([Symbol("B"), Symbol("D", negated=True), Symbol("E")])
        c2.add_symbols([Symbol("A"), Symbol("C", negated=True)])
        f.add_clauses([c1, c2])

        print(f)
        print(f.to_problem())

if __name__ == "__main__":
    TestCNF2Problem().test_cnf_to_problem()
