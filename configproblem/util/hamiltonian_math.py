from qubovert import spin_var

def solve_bruteforce(model):
    model_solution = model.solve_bruteforce()
    print("Variable assignment:", model_solution)
    print("Model value:", model.value(model_solution))
    print("Constraints satisfied?", model.is_solution_valid(model_solution)) # we don't have constraints in our model


def compute_config_energy(hamiltonian, config: str):
    """
        Computes the energy for a given configuration (ising input) of a given hamiltonian (ising form)
    """
    energy = 0
    for key, factor in hamiltonian.items():
        acting_qubits = len(key)

        if acting_qubits == 0:
            # identity case
            energy += factor
        elif acting_qubits == 1:
            # single qubit term
            q1 = key[0]
            energy += factor * config[q1]
        elif acting_qubits ==2:
            # quadratic qubit term
            q1 = key[0]
            q2 = key[1]
            energy += factor * config[q1] * config[q2]
        else:
            # non quadratic, error
            raise RuntimeError(f"Non quadratic term in hamiltonian: {key, factor}")
    return energy
                         
                         
def hamiltonian_strategy_average(hamiltonian, counts):
    """
        Computes the average energy across the entire measurement for the given hamiltonian
    """
    average_energy = 0
    for config_str in counts.keys():
        # convert the 0/1 feature string to ising integers
        config = [-1 if s == "0" else 1 for s in config_str]
        energy = compute_config_energy(hamiltonian, config)*counts.get(config_str)
        average_energy += energy
    average_energy /= counts.shots()
    return average_energy
                         

def hamiltonian_strategy_top(hamiltonian, counts):
    """
        Computes the energy of the configuration that was measured the most for the given hamiltonian
    """
    # take the config that was measured most often
    config_str = max(counts, key=counts.get) 
    # convert the 0/1 feature string to ising integers
    config = [-1 if s == "0" else 1 for s in config_str]
    return compute_config_energy(hamiltonian, config)
                        

def hamiltonian_strategy_min(hamiltonian, counts):
    """
        Finds the _measured_ configuration with the least energy and returns its value.
    """
    min_energy = float('inf')
    for config_str in counts.keys():
        # convert the 0/1 feature string to ising integers
        config = [-1 if s == "0" else 1 for s in config_str]
        energy = compute_config_energy(hamiltonian, config)
        min_energy = energy if energy < min_energy else min_energy
    return min_energy


def compute_hamiltonian_energy(hamiltonian, counts, strategy='avg'):
    """
        Compute the energy state of a hamiltonian from measurements.
        
        :param hamiltonian: the hamiltonian (QUSO) describing the system
        :param counts: measurement results for a quantum system for the hamiltonian
        :param strategy: method for actually evaluating the hamiltonian. Available: 'avg', 'top', 'min'
    """
    if strategy == 'avg':
        return hamiltonian_strategy_average(hamiltonian, counts)
    elif strategy == 'top':
        return hamiltonian_strategy_top(hamiltonian, counts)
    elif strategy == 'min':
        return hamiltonian_strategy_min(hamiltonian, counts)
    else:
        raise RuntimeError(f"Unsupported strategy: {strategy}")
