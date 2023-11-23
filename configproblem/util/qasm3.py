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
    commands: List[Tuple[Name, Qubit, List[Qubit], List[Qubit]]] = field(default_factory=list)

    def add_qubit(self, name: Name, amount: Size = 1):
        if name in self.qubit_reg:
            self.qubit_reg[name] += amount
        else:
            self.qubit_reg[name] = amount

    def add_bit(self, name: Name, amount: Size = 1):
        if name in self.bit_reg:
            self.bit_reg[name] += amount
        else:
            self.bit_reg[name] = amount

    def add_command(self, command_name: Name, tar: Qubit, controls: List[Qubit] = None, negative_controls: List[Qubit] = None,  prepend: bool = False):
        if controls is None:
            controls = []
        if negative_controls is None:
            negative_controls = []
        if prepend:
            self.commands.insert(0, (command_name, tar, controls, negative_controls))
        else:
            self.commands.append((command_name, tar, controls, negative_controls))

    def add_commands(self, commands: List[Tuple[Name, Qubit, List[Qubit], List[Qubit]]], prepend: bool = False):
        if prepend:
            for command_name, tar, controls, negative_controls in commands[::-1]:
                self.add_command(command_name, tar, controls, negative_controls, prepend=True)
        else:
            for command_name, tar, controls, negative_controls in commands:
                self.add_command(command_name, tar, controls, negative_controls)

    def width(self):
        return sum(self.qubit_reg.values())

    def calculate_depth(self):
        depths: Dict[Qubit, Size] = {(name, i): 0 for (name, size) in self.qubit_reg.items() for i in range(size)}
        for (_, tar, controls, neg_controls) in self.commands:
            new_control_depth = 0
            for nctrl in neg_controls:
                depths[nctrl] += 1
            for qubit in neg_controls + controls + [tar]:
                new_control_depth = max(new_control_depth, depths[qubit] + 1)
            for control in neg_controls + controls + [tar]:
                depths[control] = max(depths[control], new_control_depth)

            for nctrl in neg_controls:
                 depths[nctrl] += 1
        return max(depths.values())+1, depths

    def calculate_adapted_depth(self):
        depths: Dict[Qubit, Size] = {(name, i): 0 for (name, size) in self.qubit_reg.items() for i in range(size)}
        for (_, tar, controls, neg_controls) in self.commands:
            new_control_depth = 0
            # for nctrl in neg_controls:
            #     depths[nctrl] += 1
            for qubit in neg_controls + controls + [tar]:
                new_control_depth = max(new_control_depth, depths[qubit] + 1)
            for control in neg_controls + controls + [tar]:
                depths[control] = max(depths[control], new_control_depth)

            # for nctrl in  neg_controls:
            #     depths[nctrl] += 1
        return max(depths.values())+1, depths

    def to_qasm(self) -> str:
        qasm = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n"
        for qubit in self.qubit_reg:
            qasm += f"qubit[{self.qubit_reg[qubit]}] {qubit};\n"
        for bit in self.bit_reg:
            qasm += f"bit[{self.bit_reg[bit]}] {bit};\n"

        for (command, target, controls, negative_controls) in self.commands:
            if controls and not negative_controls:
                qasm += f"ctrl({len(controls)}) @ {command} {', '.join([f'{control[0]}[{control[1]}]' for control in controls])}, {target[0]}[{target[1]}];\n"
            elif controls and negative_controls:
                qasm += f"negctrl({len(negative_controls)}) @ ctrl({len(controls)}) @ {command} {', '.join([f'{negctrl[0]}[{negctrl[1]}]' for negctrl in negative_controls])}, {', '.join([f'{ctrl[0]}[{ctrl[1]}]' for ctrl in controls])}, {target[0]}[{target[1]}];\n"
            elif negative_controls:
                qasm += f"negctrl({len(negative_controls)}) @ {command} {', '.join([f'{negctrl[0]}[{negctrl[1]}]' for negctrl in negative_controls])}, {target[0]}[{target[1]}];\n"
            else:
                qasm += f"{command} {target[0]}[{target[1]}];\n"
        qasm += f"feat_bit_reg = measure feat_reg;\n"
        return qasm

