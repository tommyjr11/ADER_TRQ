from flask import Flask, render_template, request
import sympy as sp
from sympy import ccode


app = Flask(__name__)

from sympy.printing.c import C99CodePrinter


class NoPowCCodePrinter(C99CodePrinter):
    """C code printer that expands integer powers instead of using pow()."""

    def _print_Pow(self, expr):
        base, exp = expr.as_base_exp()

        # 只针对整数非负幂做展开，其余交给原来的逻辑（可能用 pow）
        if exp.is_integer and exp.is_nonnegative:
            n = int(exp)
            if n == 0:
                return "1"
            base_code = self._print(base)
            # 幂为 1 时直接返回
            if n == 1:
                return base_code
            # 复杂一点的 base 加括号更安全
            if not base.is_Atom:
                base_code = f"({base_code})"
            return " * ".join([base_code] * n)

        # 其它情况（比如非整数指数），继续用默认的实现
        return super()._print_Pow(expr)

# 一个方便调用的函数
_no_pow_printer = NoPowCCodePrinter()

def ccode_no_pow(expr):
    return _no_pow_printer.doprint(expr)




def matrix_to_cpp(M, name, type_str="double"):
    """Convert a Sympy Matrix M into a C++ 2D array declaration string.

    Example output:

    // B(3x3)
    // rows = i (0..2), cols = j (0..2)
    double B[3][3] = {
        {h0*h0, h0/2.0, 0.0},
        {h0/2.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    """
    rows, cols = M.shape
    lines = []
    lines.append(f"// {name}({rows}x{cols})")
    lines.append(f"// rows = i (0..{rows-1}), cols = j (0..{cols-1})")
    lines.append(f"{type_str} {name}[{rows}][{cols}] = {{")
    for i in range(rows):
        entries = ", ".join(ccode_no_pow(M[i, j]) for j in range(cols))
        comma = "," if i < rows - 1 else ""
        lines.append(f"    {{{entries}}}{comma}")
    lines.append("};")
    return "\n".join(lines)




def build_nodes(h_list):
    """Generate nodes: [0, -h1, -(h1+h2), ..., -(h1+...+hN)]."""
    nodes = [sp.Integer(0)]
    s = sp.Integer(0)
    for h in h_list:
        s += h
        nodes.append(-s)
    return nodes

def lagrange_basis(t, nodes):
    """Return list of L_i(t) for arbitrary nodes."""
    L = []
    for i, xi in enumerate(nodes):
        Li = sp.Integer(1)
        for j, xj in enumerate(nodes):
            if j != i:
                Li *= (t - xj) / (xi - xj)
        L.append(sp.simplify(Li))
    return L

def derivative_mapping_at_zero(L, max_m):
    """Compute D[m,k] = d^m/dt^m L_k(t) evaluated at t = 0."""
    t = sp.symbols('t', real=True)
    M = len(L)
    D = sp.Matrix.zeros(max_m + 1, M)
    for k, Lk in enumerate(L):
        for m in range(max_m + 1):
            D[m, k] = sp.simplify(sp.diff(Lk, t, m).subs(t, 0))
    return D

def build_B_matrices(t, h0, L, lmax):
    """Compute I^(l) and B = sum_l I^(l), where
       I^(l)_{ij} = ∫_0^{h0} h0^{2l-1} * (L_i^{(l)}(t)) * (L_j^{(l)}(t)) dt.
    """
    n = len(L)
    I_list = []
    B = sp.Matrix.zeros(n, n)
    for l in range(1, lmax + 1):
        # Derivatives of order l
        dL = [sp.diff(Li, t, l) for Li in L]
        I = sp.Matrix.zeros(n, n)
        kernel = lambda i, j: h0**(2*l - 1) * dL[i] * dL[j]
        for i in range(n):
            for j in range(n):
                integrand = sp.simplify(kernel(i, j))
                I[i, j] = sp.simplify(sp.integrate(integrand, (t, 0, h0)))
        I_list.append(I)
        B += I
    return I_list, sp.simplify(B)

def parse_h_series(raw):
    """Parse 'h1,h2,...' into a list of Sympy symbols/numbers."""
    parts = [p.strip() for p in raw.split(',') if p.strip() != '']
    return [sp.sympify(p) for p in parts]

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default values
    default_h0 = "h0"
    default_h_list = "h1,h2"
    default_n = "2"
    default_lmax = ""   # empty => use N
    default_m = ""      # empty => use N

    result = None
    error = None

    if request.method == 'POST':
        try:
            raw_h0 = request.form.get('h0', default_h0)
            raw_list = request.form.get('h_list', default_h_list)
            raw_n = request.form.get('n_history', default_n)
            raw_lmax = request.form.get('lmax', default_lmax)
            raw_m = request.form.get('m_deriv', default_m)

            # Convert input strings to Sympy objects
            h0 = sp.sympify(raw_h0)
            h_list = parse_h_series(raw_list)
            N = int(raw_n)

            if len(h_list) != N:
                raise ValueError(f"Number of intervals N={N}, but {len(h_list)} values were provided. Please keep them consistent.")

            lmax = int(raw_lmax) if raw_lmax.strip() else N
            M = int(raw_m) if raw_m.strip() else N
            if lmax < 1:
                raise ValueError("lmax must be ≥ 1.")
            if M < 0:
                raise ValueError("Derivative order M must be ≥ 0.")

            # Build nodes and Lagrange basis
            t = sp.symbols('t', real=True)
            nodes = build_nodes(h_list)
            L = lagrange_basis(t, nodes)

            # Compute D matrix at t = 0 (rows 0..M)
            D = derivative_mapping_at_zero(L, M)

            # Compute I^(l) and B
            I_list, B = build_B_matrices(t, h0, L, lmax)

            # Simplify all symbolic results
            L_s = [sp.simplify(li) for li in L]
            I_s = [sp.simplify(I) for I in I_list]
            B_s = sp.simplify(B)

            # result = {
            #     "nodes": nodes,
            #     "L": L_s,
            #     "I_list": I_s,
            #     "B": B_s,
            #     "D": D,
            #     "N": N,
            #     "lmax": lmax,
            #     "M": M,
            #     "h0": h0,
            #     "h_list": h_list,
            #     "repr_B": repr(B_s),      # 例如: 'Matrix([[1, 0], [0, 1]])'
            #     "repr_D": repr(D),
            # }
            result = {
                "nodes": nodes,
                "L": L_s,
                "I_list": I_s,
                "B": B_s,
                "D": D,
                "N": N,
                "lmax": lmax,
                "M": M,
                "h0": h0,
                "h_list": h_list,
                # === 新增：C++ 代码字符串 ===
                "cpp_B": matrix_to_cpp(B_s, "B"),
                "cpp_D": matrix_to_cpp(D,   "D"),
            }

        except Exception as e:
            error = str(e)

    return render_template('index.html',
                           default_h0=default_h0,
                           default_h_list=default_h_list,
                           default_n=default_n,
                           default_lmax=default_lmax,
                           default_m=default_m,
                           result=result,
                           error=error,
                           sp=sp,)

if __name__ == '__main__':
    # Run: python app.py, then open http://127.0.0.1:5000/ in browser
    app.run(debug=True)
