from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Name = str
Index = int
Size = int
Qubit = Tuple[Name, Index]
Bit = Tuple[Name, Index]


@dataclass
class QASM3:
    qubit_reg: Dict[Name, Size] = field(default_factory=dict)
    bit_reg: Dict[Name, Size] = field(default_factory=dict)
    commands: List[Tuple[Name, Qubit, List[Qubit]]] = field(default_factory=list)

    def add_qubit(self, name: Name, amount: Size = 1):
        if name in self.qubit_reg:
            self.qubit_reg[name] += amount
        else:
            self.qubit_reg[name] = amount

    def add_bit(self, name: Name, amount: Size):
        if name in self.bit_reg:
            self.bit_reg[name] += amount
        else:
            self.bit_reg[name] = amount

    def add_command(self, command_name: Name, tar: Qubit, controls: List[Qubit] = None, prepend: bool = False):
        if controls is None:
            controls = []
        if prepend:
            self.commands.insert(0, (command_name, tar, controls))
        else:
            self.commands.append((command_name, tar, controls))

    def add_commands(self, commands: List[Tuple[Name, Qubit, List[Qubit]]], prepend: bool = False):
        if prepend:
            for command_name, tar, controls in commands[::-1]:
                self.add_command(command_name, tar, controls, prepend=True)
        else:
            for command_name, tar, controls in commands:
                self.add_command(command_name, tar, controls)

    def calculate_depth(self):
        depths: Dict[Qubit, Size] = {}
        for (_, tar, controls) in self.commands:
            new_control_depth = 1
            for qubit in controls + [tar]:
                if qubit in depths:
                    new_control_depth = max(new_control_depth, depths[qubit] + 1)
                else:
                    depths[qubit] = new_control_depth
            for control in controls + [tar]:
                depths[control] = max(depths[control], new_control_depth)
        return depths

    def to_qasm(self) -> str:
        qasm = "OPENQASM 3.0;include \"stdgates.inc\";"
        for qubit in self.qubit_reg:
            qasm += f"qubit[{self.qubit_reg[qubit]}] {qubit};"
        for bit in self.bit_reg:
            qasm += f"bit[{self.bit_reg[bit]}] {bit};"

        for (command, target, controls) in self.commands:
            if controls:
                qasm += f"ctrl({len(controls)}) @ {command} {', '.join([f'{control[0]}[{control[1]}]' for control in controls])}, {target[0]}[{target[1]}];"
            else:
                qasm += f"{command} {target[0]}[{target[1]}];"
        return qasm

