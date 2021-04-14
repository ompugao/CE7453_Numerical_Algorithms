import sympy
import numpy as np

h = 9

def x_func_sympy(u):
    return (sympy.exp(sympy.cos(6.2*u + h/30)) + 0.1)*sympy.cos(12.4*u)

if __name__ == '__main__':
    u = sympy.symbols('u')
    x = x_func_sympy(u)
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

