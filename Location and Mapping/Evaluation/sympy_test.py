import numpy as np
import sympy as sym
from sympy import symbols


tx1, tz1, tx2, tz2 = symbols("tx_1 tz_1 tx_2 tz_2")
qx1, qz1, qx2, qz2 = symbols("qx_1 qz_1 qx_2 qz_2")

A = sym.Matrix([[1, 0, -tz1, tx1],
                [0, 1, tx1, tz1],
                [1, 0, -tz2, tx2],
                [0, 1, tx2, tz2]])



B = sym.Matrix([[qx1],
                [qz1],
                [qx2],
                [qz2]])

x = A.inv() * B

xt_ev = sym.simplify(x[0])
zt_ev = sym.simplify(x[1])
s_ev = sym.simplify(x[2])
c_ev = sym.simplify(x[3])