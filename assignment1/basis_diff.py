import sympy

u = sympy.symbols('u')
u0, u1, u2, u3, u4 = sympy.symbols('u:5')

expr0 = (u - u0)**3 / ((u1 - u0)*(u2-u0)*(u3-u0))
expr1 = ((u - u0)**2 * (u2 - u)) / ((u2 - u1)*(u3 - u0)*(u2 - u0)) \
    + ((u3 - u) * (u - u0) * (u - u1)) / ((u2 - u1)*(u3 - u1)*(u3 - u0)) \
    + (u4 - u) * (u - u1)**2 / ((u2 - u1)*(u4 - u1)*(u3 - u1))
expr2 = ((u - u0) * (u3 - u)**2) / ((u3 - u2)*(u3 - u1)*(u3 - u1)) \
        + ((u4 - u) * (u3 - u) * (u - u1)) / ((u3 - u2)*(u4 - u1)*(u3 - u1)) \
        + (u4 - u)**2 * (u - u2) / ((u3 - u2)*(u4 - u2)*(u4 - u1))
expr3 = (u4 - u) ** 3 / ((u4 - u3) * (u4-u2) * (u4 - u1))
print(sympy.simplify(expr0.diff(u).diff(u)))
print(sympy.simplify(expr1.diff(u).diff(u)))
print(sympy.simplify(expr2.diff(u).diff(u)))
print(sympy.simplify(expr3.diff(u).diff(u)))

# 6*(-u + u0)/((u0 - u1)*(u0 - u2)*(u0 - u3))
# 2*((u0 - u2)*(u0 - u3)*(3*u - 2*u1 - u4) + (u0 - u2)*(u1 - u4)*(3*u - u0 - u1 - u3) + (u1 - u3)*(u1 - u4)*(3*u - 2*u0 - u2))/((u0 - u2)*(u0 - u3)*(u1 - u2)*(u1 - u3)*(u1 - u4))
# 2*((u1 - u3)**2*(-3*u + u2 + 2*u4) + (u1 - u3)*(u2 - u4)*(-3*u + u1 + u3 + u4) + (u1 - u4)*(u2 - u4)*(-3*u + u0 + 2*u3))/((u1 - u3)**2*(u1 - u4)*(u2 - u3)*(u2 - u4))
# 6*(u - u4)/((u1 - u4)*(u2 - u4)*(u3 - u4))
from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())


