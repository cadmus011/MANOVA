import numpy as np
from scipy.stats import f as f_dist


def get_user_data():
    """Collects group observations interactively or loads the Slide 18 example."""
    print("Choose an option:")
    print("1. Enter your own data")
    print("2. Run the built-in example from Slide 18 (3 methods, 2 DVs, n=5 each)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
    
        groups = [
            np.array([[6, 8], [7, 6], [8, 9], [5, 7], [9, 10]], dtype=float),
            np.array([[3, 9], [4, 7], [5, 10], [2, 8], [6, 11]], dtype=float),
            np.array([[4, 4], [3, 6], [5, 5], [2, 3], [6, 7]], dtype=float),
        ]
        group_names = ["Method A", "Method B", "Method C"]
        return groups, group_names

    g = int(input("\nEnter the number of groups (g): "))
    p = int(input("Enter the number of dependent variables (p): "))

    groups = []
    group_names = []

    for i in range(g):
        name = input(f"\nEnter name for Group {i + 1} (or press Enter for 'Group {i + 1}'): ").strip()
        if not name:
            name = f"Group {i + 1}"
        group_names.append(name)

        n_i = int(input(f"Enter number of observations in {name}: "))
        print(f"Enter {n_i} rows with {p} space-separated values each:")

        rows = []
        for j in range(n_i):
            while True:
                try:
                    vals = list(map(float, input(f"  Obs {j + 1}: ").strip().split()))
                    if len(vals) != p:
                        print(f"  Please enter exactly {p} numbers.")
                        continue
                    rows.append(vals)
                    break
                except ValueError:
                    print("  Invalid input. Enter numeric values separated by spaces.")
        groups.append(np.array(rows, dtype=float))

    return groups, group_names


def compute_manova(groups):
    """Computes H, E, T matrices, eigenvalues of E^-1 H, and the 4 MANOVA test statistics."""
    g = len(groups)
    p = groups[0].shape[1]
    all_data = np.vstack(groups)
    N = all_data.shape[0]
    grand_mean = np.mean(all_data, axis=0)

    
    H = np.zeros((p, p))
    E = np.zeros((p, p))

    for grp in groups:
        n_j = grp.shape[0]
        grp_mean = np.mean(grp, axis=0)

        diff_between = (grp_mean - grand_mean).reshape(-1, 1)
        H += n_j * (diff_between @ diff_between.T)

        diff_within = grp - grp_mean
        E += diff_within.T @ diff_within

    T = H + E

    E_inv_H = np.linalg.inv(E) @ H
    raw_eigvals = np.linalg.eigvals(E_inv_H)
    s = min(p, g - 1)
    eigvals = np.sort(np.real(raw_eigvals))[::-1][:s]

    wilks_lambda = np.prod(1.0 / (1.0 + eigvals))
    pillais_trace = np.sum(eigvals / (1.0 + eigvals))
    hotelling_lawley = np.sum(eigvals)
    roys_lambda1 = eigvals[0]                 # Reported by SPSS
    roys_theta = roys_lambda1 / (1.0 + roys_lambda1)

    df_h = g - 1
    df_e = N - g
    m = (abs(p - df_h) - 1) / 2.0
    n_param = (df_e - p - 1) / 2.0

    if p * df_h == 2:
        t_wilks = 1.0
    else:
        num = (p**2 * df_h**2 - 4)
        den = (p**2 + df_h**2 - 5)
        t_wilks = np.sqrt(num / den) if den > 0 else 1.0

    r_wilks = df_e - (p - df_h + 1) / 2.0
    u_wilks = (p * df_h - 2) / 4.0
    w_df1 = p * df_h
    w_df2 = r_wilks * t_wilks - 2 * u_wilks
    w_root = wilks_lambda ** (1.0 / t_wilks)
    w_F = ((1.0 - w_root) / w_root) * (w_df2 / w_df1)
    w_p = 1.0 - f_dist.cdf(w_F, w_df1, w_df2)
    w_eta2 = 1.0 - (wilks_lambda ** (1.0 / s))

    p_df1 = s * (2 * m + s + 1)
    p_df2 = s * (2 * n_param + s + 1)
    p_F = ((2 * n_param + s + 1) / (2 * m + s + 1)) * (pillais_trace / (s - pillais_trace))
    p_p = 1.0 - f_dist.cdf(p_F, p_df1, p_df2)
    p_eta2 = pillais_trace / s
    
    hl_df1 = s * (2 * m + s + 1)
    hl_df2 = 2 * (s * n_param + 1)
    hl_F = (hl_df2 / (s * hl_df1)) * hotelling_lawley
    hl_p = 1.0 - f_dist.cdf(hl_F, hl_df1, hl_df2)
    hl_eta2 = (hotelling_lawley / s) / (1.0 + (hotelling_lawley / s))

    r_df1 = max(p, df_h)
    r_df2 = df_e - r_df1 + df_h
    r_F = roys_lambda1 * (r_df2 / r_df1)
    r_p = 1.0 - f_dist.cdf(r_F, r_df1, r_df2)
    r_eta2 = roys_theta

    return {
        "N": N, "g": g, "p": p, "s": s,
        "H": H, "E": E, "T": T, "eigvals": eigvals,
        "tests": [
            ("Pillai's Trace", pillais_trace, p_F, p_df1, p_df2, p_p, p_eta2),
            ("Wilks' Lambda", wilks_lambda, w_F, w_df1, w_df2, w_p, w_eta2),
            ("Hotelling-Lawley Trace", hotelling_lawley, hl_F, hl_df1, hl_df2, hl_p, hl_eta2),
            ("Roy's Largest Root (λ1)", roys_lambda1, r_F, r_df1, r_df2, r_p, r_eta2),
        ],
        "roys_theta": roys_theta,
    }


def print_results(res):
    print("\n" + "=" * 78)
    print(f"MANOVA RESULTS  (N = {res['N']}, Groups g = {res['g']}, DVs p = {res['p']}, s = {res['s']})")
    print("=" * 78)

    np.set_printoptions(precision=3, suppress=True)
    print("\nBetween-Groups SSCP Matrix (H):\n", res["H"])
    print("\nWithin-Groups SSCP Matrix (E):\n", res["E"])
    print("\nTotal SSCP Matrix (T = H + E):\n", res["T"])
    print("\nEigenvalues of E^-1 H:", np.round(res["eigvals"], 4))

    print("\n" + "-" * 78)
    print(f"{'Test Statistic':<26} {'Value':>8} {'Approx F':>10} {'Hyp df':>8} {'Err df':>8} {'p-value':>10} {'Part η²':>8}")
    print("-" * 78)
    for name, val, F_val, df1, df2, p_val, eta2 in res["tests"]:
        p_str = "<.001" if p_val < 0.001 else f"{p_val:.4f}"
        print(f"{name:<26} {val:>8.4f} {F_val:>10.3f} {df1:>8.1f} {df2:>8.1f} {p_str:>10} {eta2:>8.3f}")
    print("-" * 78)
    print(f"Note: Roy's Largest Root θ = λ1 / (1 + λ1) = {res['roys_theta']:.4f}")


if __name__ == "__main__":
    groups, _ = get_user_data()
    results = compute_manova(groups)
    print_results(results)