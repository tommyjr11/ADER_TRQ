from flask import Flask, render_template, request
import sympy as sp

app = Flask(__name__)

# ----------------  计算 WENO 权重的核心函数 ----------------
def compute_weno_weights(r=5, eval_x=sp.Rational(1, 2)):
    U = {i: sp.symbols(f"U{i}") for i in range(-(r - 1), r)}
    x = sp.Symbol("x")

    # 构造单个小模板的重构多项式
    def reconstruct_stencil(cells):
        coeffs = sp.symbols(f"a0:{r}")
        P = sum(coeffs[k] * x**k for k in range(r))
        eqs = [sp.integrate(P, (x, j, j + 1)) - U[j] for j in cells]
        sol = sp.solve(eqs, coeffs, rational=True)
        return sp.expand(P.subs(sol).subs(x, eval_x))

    # r 个左右滑动模板
    stencils = [list(range(-(r - 1) + s, s + 1)) for s in range(r)]
    Q_stencils = [reconstruct_stencil(st) for st in stencils]

    # 中央模板多项式（阶数 2r−2）
    deg = 2 * r - 2
    coeffs_central = sp.symbols(f"A0:{deg + 1}")
    P_central = sum(coeffs_central[k] * x**k for k in range(deg + 1))
    eqs_central = [
        sp.integrate(P_central, (x, j, j + 1)) - U[j] for j in range(-(r - 1), r)
    ]
    sol_central = sp.solve(eqs_central, coeffs_central, rational=True)
    Q_central = sp.expand(P_central.subs(sol_central).subs(x, eval_x))

    # 把多项式对 U_i 的偏导数当作系数向量
    coeff_indices = list(range(-(r - 1), r))

    def coeff_vector(expr):
        return [sp.diff(expr, U[i]) for i in coeff_indices]

    C_central = coeff_vector(Q_central)
    C_stencils = [coeff_vector(q) for q in Q_stencils]

    # 解权重 w_j，使得加权组合等于中央模板
    w = sp.symbols(f"w0:{r}")
    eqs_weights = [
        sum(w[j] * C_stencils[j][k] for j in range(r)) - C_central[k]
        for k in range(len(coeff_indices))
    ]
    sol_weights = sp.solve(eqs_weights, w, rational=True)
    return sol_weights


# ----------------          路由            ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    # 默认值
    weights = None
    r, eval_x_str = 5, "1/2"
    if request.method == "POST":
        r = int(request.form.get("r", 5))
        eval_x_str = request.form.get("eval_x", "1/2")
        try:
            eval_x = sp.sympify(eval_x_str)
        except Exception:
            eval_x = sp.Rational(eval_x_str)
        try:
            weights = compute_weno_weights(r, eval_x)
        except Exception as e:
            # 计算失败时传递错误信息
            weights = {"error": str(e)}

    return render_template(
        "index.html",
        weights=weights,
        r=r,
        eval_x=eval_x_str,
    )



from flask import Flask, render_template, request, redirect, url_for
import sympy as sp

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# 1.  WENO weight routine (unchanged, still served at "/")
# ─────────────────────────────────────────────────────────────
def compute_weno_weights(r=5, eval_x=sp.Rational(1, 2)):
    U = {i: sp.symbols(f"U{i}") for i in range(-(r - 1), r)}
    x = sp.Symbol("x")

    def reconstruct_stencil(cells):
        coeffs = sp.symbols(f"a0:{r}")
        P = sum(coeffs[k] * x**k for k in range(r))
        sol = sp.solve(
            [sp.integrate(P, (x, j, j + 1)) - U[j] for j in cells], coeffs, rational=True
        )
        return sp.expand(P.subs(sol).subs(x, eval_x))

    stencils = [list(range(-(r - 1) + s, s + 1)) for s in range(r)]
    Q_stencils = [reconstruct_stencil(st) for st in stencils]

    deg = 2 * r - 2
    coeffs_central = sp.symbols(f"A0:{deg + 1}")
    P_central = sum(coeffs_central[k] * x**k for k in range(deg + 1))
    sol_central = sp.solve(
        [
            sp.integrate(P_central, (x, j, j + 1)) - U[j]
            for j in range(-(r - 1), r)
        ],
        coeffs_central,
        rational=True,
    )
    Q_central = sp.expand(P_central.subs(sol_central).subs(x, eval_x))

    coeff_indices = list(range(-(r - 1), r))

    def coeff_vec(expr):
        return [sp.diff(expr, U[i]) for i in coeff_indices]

    C_central = coeff_vec(Q_central)
    C_stencils = [coeff_vec(q) for q in Q_stencils]

    w = sp.symbols(f"w0:{r}")
    sol_w = sp.solve(
        [
            sum(w[j] * C_stencils[j][k] for j in range(r)) - C_central[k]
            for k in range(len(coeff_indices))
        ],
        w,
        rational=True,
    )
    return sol_w


# ─────────────────────────────────────────────────────────────
# 2.  Generic polynomial reconstruction routine (served at "/poly")
# ─────────────────────────────────────────────────────────────
def reconstruct_poly(r=5, eval_x=sp.Rational(1, 2)):
    """
    Build a polynomial of degree r-1 whose cell averages on [j,j+1]
    equal symbols u_j, j = 0..r-1, then return P(eval_x) and all
    derivatives up to order r-1.
    """
    if r < 2:
        raise ValueError("r must be ≥ 2")

    # coefficients a0 … a_{r-1}
    coeffs = sp.symbols(f"a0:{r}")
    x = sp.Symbol("x")
    P = sum(coeffs[k] * x**k for k in range(r))

    # symbols for cell averages
    u = sp.symbols(f"u0:{r}")

    # equations: ∫_{j}^{j+1} P dx = u_j
    equations = [
        sp.integrate(P, (x, j, j + 1)) - u[j] for j in range(r)
    ]

    sol_coeffs = sp.solve(equations, coeffs, simplify=True)

    # substitute coefficients into polynomial
    P_sub = P.subs(sol_coeffs)

    # build derivative list 0 … r-1
    derivs = [sp.diff(P_sub, x, k).subs(x, eval_x).simplify() for k in range(r)]

    # pretty keys: Q, Qx, Qxx, …
    names = ["Q"] + [f"Q{ 'x'*k }" for k in range(1, r)]
    return dict(zip(names, derivs))


# ─────────────────────────────────────────────────────────────
# 3.  Routes
# ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    weights = None
    r, eval_x_str = 5, "1/2"
    if request.method == "POST":
        r = int(request.form.get("r", 5))
        eval_x_str = request.form.get("eval_x", "1/2")
        eval_x = sp.sympify(eval_x_str)
        try:
            weights = compute_weno_weights(r, eval_x)
        except Exception as e:
            weights = {"error": str(e)}

    return render_template(
        "index.html", weights=weights, r=r, eval_x=eval_x_str
    )


@app.route("/poly", methods=["GET", "POST"])
def poly():
    results = None
    r, eval_x_str = 5, "1/2"
    if request.method == "POST":
        r = int(request.form.get("r", 5))
        eval_x_str = request.form.get("eval_x", "1/2")
        try:
            eval_x = sp.sympify(eval_x_str)
            results = reconstruct_poly(r, eval_x)
        except Exception as e:
            results = {"error": str(e)}

    return render_template(
        "poly.html", results=results, r=r, eval_x=eval_x_str
    )


# optional: redirect "/reconstruct" → "/poly"
@app.route("/reconstruct")
def redirect_poly():
    return redirect(url_for("poly"), code=302)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=80)



