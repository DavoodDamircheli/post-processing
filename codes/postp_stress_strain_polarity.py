#!/usr/bin/env python3
"""
postprocess_stress_strain.py

Extracts stress and strain from DEM/peridynamic-style HDF5 outputs and computes TWO stress paths:
  A) "Virial-like" stress from nodal forces (per particle) + reaction-based global stresses
  B) Constitutive stress from strain using user-provided elastic constants

Per timestep inputs:
- tc_XXXXX.h5
    /P_xxxxx/CurrPos  [Ni,3]
    /P_xxxxx/force    [Ni,3]       (for virial stress)
    /P_xxxxx/volume   [scalar]     (optional)
- wall_XXXXX.h5
    /wall_info  [6,1]   -> [x_min,y_min,z_min,x_max,y_max,z_max]
    /reaction   [6,3]   -> forces on faces (for global stress)
Reference geometry:
- setup.h5 (or all.h5)
    /P_xxxxx/Pos   [Ni,3]

Outputs:
- global_summary.csv
- per_particle_strain.csv
- per_particle_stress.csv  (includes BOTH virial and constitutive stresses)

Constitutive models:
- "linear" (small strain): eps = sym(grad u) ≈ 0.5(F + F^T) - I;   sigma = λ tr(eps) I + 2 μ eps
- "stvk"   (finite strain): E = 0.5(F^T F - I);  S = λ tr(E) I + 2 μ E;   σ = (1/J) F S F^T

Volume options for virial stress:
- "auto":    prefer dataset /volume, else convex hull of CurrPos (needs SciPy)
- "dataset": require /volume
- "hull":    convex hull volume
- "shell":   spherical shell with per-particle radii datasets if available, else global shell radii

Set paths, material properties, and options in the __main__ section.
"""

import os, re, glob, csv
import h5py
import numpy as np
# --- add near the imports ---
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


# --------------------------- File helpers ---------------------------
def is_valid_hdf5(path, min_bytes=512):
    """Fast sanity check: size and HDF5 magic."""
    try:
        st = os.stat(path)
        if not os.path.isfile(path) or st.st_size < min_bytes:
            return False
        with open(path, "rb") as fh:
            sig = fh.read(8)
        return sig == HDF5_MAGIC
    except FileNotFoundError:
        return False

def iter_valid_steps(data_dir):
    """Yield (t, tc_file, wall_file) only for valid pairs."""
    steps = list_timesteps(data_dir)  # as in your code
    for t, tc_file in steps:
        # Normalize to avoid stray whitespace/hidden chars
        tc_file = os.path.abspath(str(tc_file).strip())
        wall_file = os.path.abspath(os.path.join(data_dir, f"wall_{t:05d}.h5"))
        reasons = []
        if not os.path.exists(wall_file):
            reasons.append("missing wall file")
        if not is_valid_hdf5(tc_file):
            reasons.append("invalid/corrupt HDF5")
        if reasons:
            print(f"WARNING: skipping t={t}  tc='{tc_file}'  wall='{wall_file}'  -> {', '.join(reasons)}")
            continue
        yield t, tc_file, wall_file


def apply_strain_convention(eps, convention="compression_positive"):
    if convention == "compression_positive":
        return -eps
    return eps

def groups_like_particles(h5file):
    return [k for k, v in h5file.items() if isinstance(v, h5py.Group) and k.startswith("P_")]

def decide_force_polarity_for_timestep(tc_file, top_frac=0.5, min_nodes=2000):
    """
    Return +1 to keep forces, -1 to flip, based on mean of top-|r·f|.
    Do this ONCE per timestep file; apply to all particles for that step.
    """
    all_dots = []
    with h5py.File(tc_file, "r") as f:
        for pname in groups_like_particles(f):
            g = f[pname]
            if "CurrPos" not in g or "force" not in g: 
                continue
            x  = np.array(g["CurrPos"], dtype=float)
            fi = np.array(g["force"],   dtype=float)
            if x.shape != fi.shape or x.size == 0:
                continue
            ri = x - x.mean(axis=0, keepdims=True)
            dots = np.einsum("ij,ij->i", ri, fi)  # r·f
            if dots.size:
                k = max(1, int(top_frac * dots.size))
                idx = np.argsort(np.abs(dots))[-k:]
                all_dots.append(dots[idx])
    if not all_dots:
        return +1  # default
    dots = np.concatenate(all_dots)
    # Optional: guard against very low load noise
    if dots.size < min_nodes and np.abs(dots).mean() < 1e-12:
        return +1
    return +1 if dots.mean() > 0.0 else -1


def list_timesteps(data_dir):
    tc_files = sorted(glob.glob(os.path.join(data_dir, "tc_*.h5")))
    steps = []
    for f in tc_files:
        m = re.search(r"tc_(\d+)\.h5$", os.path.basename(f))
        if m:
            steps.append((int(m.group(1)), f))
    return steps

def read_wall(wall_file):
    with h5py.File(wall_file, "r") as w:
        ss = np.array(w["wall_info"])[:,0] if w["wall_info"].ndim == 2 else np.array(w["wall_info"]).ravel()
        if ss.size != 6:
            raise ValueError(f"/wall_info must have 6 values in {wall_file}")
        winfo = dict(x_min=float(ss[0]), y_min=float(ss[1]), z_min=float(ss[2]),
                     x_max=float(ss[3]), y_max=float(ss[4]), z_max=float(ss[5]))
        react = np.array(w["reaction"]) if "reaction" in w else None
        return winfo, react


def read_reference_positions(setup_file):
     ref = {}
     with h5py.File(setup_file, "r") as f:
        for name in f:
            # match P_00000, P_00001, ...
            if re.match(r"P_\d+$", name) and "Pos" in f[name]:
                ref[name] = np.array(f[name]["Pos"])
     return ref
 
def groups_like_particles(h5file):
     for name in h5file.keys():
        if re.match(r"P_\d+$", name):
             yield name



# --------------------------- Volume helpers ---------------------------
def particle_volume_from_dataset(g):
    if "volume" in g:
        return float(np.array(g["volume"]).squeeze())
    return None

def particle_volume_from_hull(currpos):
    if currpos.ndim != 2 or currpos.shape[1] != 3 or currpos.shape[0] < 4:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(currpos)
        return float(hull.volume)
    except Exception:
        return 0.0

def particle_volume_from_shell(g, global_outer, global_inner):
    outer_keys = ["r_outer", "radius_outer", "outer_radius", "R_outer", "Rout"]
    inner_keys = ["r_inner", "radius_inner", "inner_radius", "R_inner", "Rin"]
    R_outer = None; R_inner = None
    for k in outer_keys:
        if k in g: R_outer = float(np.array(g[k]).squeeze()); break
    for k in inner_keys:
        if k in g: R_inner = float(np.array(g[k]).squeeze()); break
    if R_outer is None: R_outer = global_outer
    if R_inner is None: R_inner = global_inner
    return (4.0/3.0) * np.pi * (R_outer**3 - R_inner**3)

def get_particle_volume(g, currpos, method="auto", shell_outer=None, shell_inner=None):
    if method in ("dataset", "auto"):
        v = particle_volume_from_dataset(g)
        if v is not None: return v, "dataset"
    if method in ("hull", "auto"):
        v = particle_volume_from_hull(currpos)
        return v, "hull"
    if method == "shell":
        if shell_outer is None or shell_inner is None:
            raise ValueError("shell method requires shell_outer and shell_inner")
        v = particle_volume_from_shell(g, shell_outer, shell_inner)
        return v, "shell"
    raise ValueError(f"Unknown volume method: {method}")

# --------------------------- Kinematics ---------------------------
def best_fit_F(X, x):
    Xc = X.mean(axis=0); xc = x.mean(axis=0)
    dX = X - Xc; dx = x - xc
    B = dX.T @ dX
    A = dx.T @ dX
    if np.linalg.cond(B) > 1e12:
        B = B + 1e-12*np.eye(3)
    F = A @ np.linalg.inv(B)
    return F, Xc, xc

def green_lagrange_E(F):
    C = F.T @ F
    return 0.5 * (C - np.eye(3))

def small_strain_eps_from_F(F):
    return 0.5 * (F + F.T) - np.eye(3)

def von_mises_from_tensor(T):
    tr = np.trace(T)
    S = T - (tr/3.0)*np.eye(3)
    return float(np.sqrt(1.5*np.sum(S*S)))

# --------------------------- Materials: resolve λ, μ ---------------------------
def resolve_lame(E=None, nu=None, K=None, G=None, lam=None, mu=None):
    """
    Resolve Lamé parameters (lam, mu) from any consistent pair.
    Priority: (lam,mu) > (E,nu) > (K,G) > (E,G) > (E,K).
    """
    # Direct
    if lam is not None and mu is not None:
        return float(lam), float(mu)

    # From E, nu
    if E is not None and nu is not None:
        E = float(E); nu = float(nu)
        mu = E/(2*(1+nu))
        lam = E*nu/((1+nu)*(1-2*nu))
        return lam, mu

    # From K, G
    if K is not None and G is not None:
        K = float(K); G = float(G)
        mu = G
        lam = K - 2*G/3.0
        return lam, mu

    # From E, G
    if E is not None and G is not None:
        E = float(E); G = float(G)
        nu = E/(2*G) - 1.0
        mu = G
        lam = 2*G*nu/(1-2*nu)
        return lam, mu

    # From E, K
    if E is not None and K is not None:
        E = float(E); K = float(K)
        G = 3*K*E/(9*K - E)
        mu = G
        lam = K - 2*G/3.0
        return lam, mu

    raise ValueError("Insufficient or inconsistent material data to resolve (lambda, mu). Provide (E,nu) or (K,G) or (lam,mu) etc.")

# --------------------------- Global stress/strain from walls ---------------------------
def global_engineering_strain(w0, w):
    L0 = np.array([w0["x_max"]-w0["x_min"], w0["y_max"]-w0["y_min"], w0["z_max"]-w0["z_min"]])
    L  = np.array([w["x_max"] -w["x_min"],  w["y_max"] -w["y_min"],  w["z_max"] -w["z_min"]])
    eps = (L - L0)/L0
    return {"ex": float(eps[0]), "ey": float(eps[1]), "ez": float(eps[2])}

def global_stress_from_reaction(w, reaction):
    Lx = w["x_max"]-w["x_min"]; Ly = w["y_max"]-w["y_min"]; Lz = w["z_max"]-w["z_min"]
    A_yz = Ly*Lz; A_xz = Lx*Lz; A_xy = Lx*Ly
    # assumed order: [x_min, y_min, z_min, x_max, y_max, z_max]
    sigma_x = (reaction[3][0] - reaction[0][0])/(2*A_yz) if A_yz>0 else np.nan
    sigma_y = (reaction[4][1] - reaction[1][1])/(2*A_xz) if A_xz>0 else np.nan
    sigma_z = (reaction[5][2] - reaction[2][2])/(2*A_xy) if A_xy>0 else np.nan
    return {"sx_reaction": float(sigma_x), "sy_reaction": float(sigma_y), "sz_reaction": float(sigma_z)}

# --------------------------- Constitutive stresses ---------------------------
def constitutive_stress_from_F(F, lam, mu, model="linear"):
    """
    Returns Cauchy stress from F using either:
      model="linear": small-strain Hooke (Cauchy)
      model="stvk"  : St. Venant–Kirchhoff; compute 2nd PK S then push-forward to Cauchy
    """
    if model == "linear":
        eps = small_strain_eps_from_F(F)
        sigma = lam*np.trace(eps)*np.eye(3) + 2*mu*eps
        return sigma, eps, None  # (sigma, small-strain eps, E_GL=None)
    elif model == "stvk":
        E = green_lagrange_E(F)
        S = lam*np.trace(E)*np.eye(3) + 2*mu*E   # 2nd PK
        J = np.linalg.det(F)
        if J <= 0:
            # Degenerate; return NaNs
            return np.full((3,3), np.nan), None, E
        sigma = (1.0/J) * (F @ S @ F.T)          # Cauchy
        return sigma, None, E
    else:
        raise ValueError("model must be 'linear' or 'stvk'")

# --------------------------- Per-timestep processing ---------------------------
# --------------------------- Per-timestep processing ---------------------------
def process_timestep(tc_file, wall_file, ref_pos, vol_method, shell_outer, shell_inner, lam, mu, constitutive_model):
    w, reaction = read_wall(wall_file)
    timestep = int(re.search(r"tc_(\d+)\.h5$", os.path.basename(tc_file)).group(1))

    pstrain_rows = []
    pstress_rows = []

    # decide polarity ONCE for this timestep file
    pol = decide_force_polarity_for_timestep(tc_file, top_frac=0.5)
    print(f"[virial] {os.path.basename(tc_file)}: polarity={pol:+d} ({'keep' if pol>0 else 'flip'})")

    with h5py.File(tc_file, "r") as f:
        for pname in groups_like_particles(f):
            if pname not in ref_pos:
                continue
            g = f[pname]
            X = ref_pos[pname]
            x = np.array(g["CurrPos"])

            # Kinematics
            F, Xc, xc = best_fit_F(X, x)
            E_GL = green_lagrange_E(F)
            E_GL_vM = von_mises_from_tensor(E_GL)

            # Virial stress (needs forces + volume)
            Vp, v_method = get_particle_volume(
                g, x, method=vol_method,
                shell_outer=shell_outer, shell_inner=shell_inner
            )
            if "force" in g and Vp and Vp > 0:
                fi_raw = np.array(g["force"], dtype=float)
                fi = pol * fi_raw  # apply global polarity for this timestep
                ri = x - x.mean(axis=0)

                Sigma_v = -(ri.T @ fi) / Vp
                Sigma_v = 0.5 * (Sigma_v + Sigma_v.T)
                SvM = von_mises_from_tensor(Sigma_v)
            else:
                Sigma_v = np.full((3, 3), np.nan)
                SvM = np.nan

            # (… then continue with constitutive stress + append rows)



# Constitutive stress from strain
            Sigma_c, eps_small, E_used = constitutive_stress_from_F(F, lam, mu, model=constitutive_model)
            SvM_c = von_mises_from_tensor(Sigma_c) if np.all(np.isfinite(Sigma_c)) else np.nan

            # Per-particle strain row
            pstrain_rows.append({
                "timestep": timestep, "particle": pname,
                "F11": F[0,0], "F12": F[0,1], "F13": F[0,2],
                "F21": F[1,0], "F22": F[1,1], "F23": F[1,2],
                "F31": F[2,0], "F32": F[2,1], "F33": F[2,2],
                "E11": E_GL[0,0], "E22": E_GL[1,1], "E33": E_GL[2,2],
                "E12": E_GL[0,1], "E13": E_GL[0,2], "E23": E_GL[1,2],
                "E_von_mises": E_GL_vM
            })

            # Per-particle stress row includes BOTH virial and constitutive
            row = {
                "timestep": timestep, "particle": pname,
                # Virial/Cauchy
                "S11_virial": Sigma_v[0,0], "S22_virial": Sigma_v[1,1], "S33_virial": Sigma_v[2,2],
                "S12_virial": Sigma_v[0,1], "S13_virial": Sigma_v[0,2], "S23_virial": Sigma_v[1,2],
                "SvonM_virial": SvM,
                "volume": Vp, "volume_method": v_method,
                # Constitutive/Cauchy
                "S11_const": Sigma_c[0,0], "S22_const": Sigma_c[1,1], "S33_const": Sigma_c[2,2],
                "S12_const": Sigma_c[0,1], "S13_const": Sigma_c[0,2], "S23_const": Sigma_c[1,2],
                "SvonM_const": SvM_c,
                "constitutive_model": constitutive_model,
                "lambda": lam, "mu": mu
            }
            # Optionally include small-strain components if model="linear"
            if eps_small is not None:
                row.update({
                    "eps11_small": eps_small[0,0], "eps22_small": eps_small[1,1], "eps33_small": eps_small[2,2],
                    "eps12_small": eps_small[0,1], "eps13_small": eps_small[0,2], "eps23_small": eps_small[1,2]
                })
            pstress_rows.append(row)

    # Global row
    global_row = {"timestep": timestep,
                  "x_min": w["x_min"], "y_min": w["y_min"], "z_min": w["z_min"],
                  "x_max": w["x_max"], "y_max": w["y_max"], "z_max": w["z_max"]}
    if reaction is not None:
        global_row.update(global_stress_from_reaction(w, reaction))

    return global_row, pstrain_rows, pstress_rows, w

# --------------------------- Driver ---------------------------
def run(data_dir, setup_file, output_dir,
        volume_method="auto", shell_outer=None, shell_inner=None,
        # Material inputs (any consistent pair or (lam,mu))
        E=None, nu=None, K=None, G=None, lam=None, mu=None,
        constitutive_model="linear"):
    os.makedirs(output_dir, exist_ok=True)

    # Resolve Lamé
    lam, mu = resolve_lame(E=E, nu=nu, K=K, G=G, lam=lam, mu=mu)

    steps = list_timesteps(data_dir)
    if not steps:
        raise RuntimeError(f"No tc_*.h5 found in {data_dir}")
    ref_pos = read_reference_positions(setup_file)
    

    global_rows = []
    per_particle_strain_all = []
    per_particle_stress_all = []

    # Reference wall for global engineering strains
    first_t, _ = steps[0]
    first_wall = os.path.join(data_dir, f"wall_{first_t:05d}.h5")
    w0, _ = read_wall(first_wall)

    #for t, tc_file in steps:
    for t, tc_file, wall_file in iter_valid_steps(data_dir):
        #wall_file = os.path.join(data_dir, f"wall_{t:05d}.h5")
        if not os.path.exists(wall_file):
            continue

        grow, strain_rows, stress_rows, w = process_timestep(
            tc_file, wall_file, ref_pos,
            volume_method, shell_outer, shell_inner,
            lam, mu, constitutive_model
        )

        # Add global engineering strain
        eps = global_engineering_strain(w0, w)
        grow.update(eps)

        global_rows.append(grow)
        per_particle_strain_all.extend(strain_rows)
        per_particle_stress_all.extend(stress_rows)

    # Write CSVs
    global_csv = os.path.join(output_dir, "global_summary.csv")
    pstrain_csv = os.path.join(output_dir, "per_particle_strain.csv")
    pstress_csv = os.path.join(output_dir, f"per_particle_stress_{constitutive_model}.csv")

    with open(global_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(global_rows[0].keys()))
        writer.writeheader()
        writer.writerows(global_rows)

    with open(pstrain_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_particle_strain_all[0].keys()))
        writer.writeheader()
        writer.writerows(per_particle_strain_all)

    with open(pstress_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_particle_stress_all[0].keys()))
        writer.writeheader()
        writer.writerows(per_particle_stress_all)

    print(f"Wrote:\n  {global_csv}\n  {pstrain_csv}\n  {pstress_csv}")

if __name__ == "__main__":
    


    # --------- singel spheres ---------
    
    data_dir   = "/home/davood/projects/beta_perigrain_v2/examples_output/compress/single_sphere/N1R12/"
    output_dir ="/home/davood/Downloads/3d-paper-results/postprocessing/single_sphere/thick-Aug28-1/csv/" 
    # output_dir = "/home/davood/Downloads/3d-paper-results/postprocessing/single_sphere/thin_N2_aug18_2"
    # setup_file = "/home/davood/Downloads/3d-paper-results/single_sphere/master-thin-N2/setup.h5" # contains /P_xxxxx/Pos
    #
    shell_outer = 200e-3            # used if volume_method == "shell"
    shell_inner = (1/2) * shell_outer

   


    ''' 
    
    # ---------Aggregate-------------
    data_dir="/media/davood/093c4011-b7d0-4917-a86a-7c2fb7e4c748/project_data/force-trace/3d-ordinary-shpes/new_setup_hollow_sphere/N400"
    #data_dir = "/media/davood/093c4011-b7d0-4917-a86a-7c2fb7e4c748/project_data/result-periDEM/compress/single-sphere/N2"
    output_dir = "/home/davood/Downloads/3d-paper-results/postprocessing/multi_sphere_compress/R07_N400_aug24_1"
    '''


    #data_dir = "/media/davood/093c4011-b7d0-4917-a86a-7c2fb7e4c748/project_data/result-periDEM/compress/single-sphere/N2"
    setup_file = os.path.join(data_dir, "setup.h5")
    #output_dir = "/home/davood/Downloads/3d-paper-results/postprocessing/single_sphere/thin_N2_aug19_1"
    # Volume for virial stress
    volume_method = "shell"        # "auto" | "dataset" | "hull" | "shell"
    #shell_outer = 1e-3            # used if volume_method == "shell"
    #shell_inner = 0.7 * shell_outer 


    # Material properties (provide any consistent pair or lam,mu)
    bulk_modulus = 2.e+09
    shear_modulus = 1.33e+09
    rho=1200
    # Poisson's ratio [-]
    nu = (3 * bulk_modulus - 2 * shear_modulus) / ( 2 * ( 3 * bulk_modulus + shear_modulus))
           # Young's modulus [Pa]
    E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus) 


    K  = None      # Bulk modulus [Pa]
    G  = None      # Shear modulus [Pa]
    lam = None     # Lamé lambda [Pa]
    mu  = None     # Lamé mu (shear) [Pa]

    # Choose constitutive model for "stress from strain"
    #   "linear": small-strain Hooke (Cauchy)
    #   "stvk"  : St. Venant–Kirchhoff (finite strain); outputs Cauchy via push-forward
    constitutive_model = "stvk"

    run(data_dir, setup_file, output_dir,
        volume_method, shell_outer, shell_inner,
        E, nu, K, G, lam, mu,
        constitutive_model)
