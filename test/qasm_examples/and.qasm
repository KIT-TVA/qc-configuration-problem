OPENQASM 3.0;
include ("stdgates.qasm")

qubit[3] inp_reg;
bit cont;

ctrl(3) @ x inp_reg, cont;