#!/usr/bin/env python3
"""
Standalone post-processing with extended plots (v2, dual stress paths).

Enhancements:
- Compute per-particle and aggregated stress invariants for BOTH
  (a) force-based virial stress columns:  Sij_virial
  (b) strain-based stress columns:       Sij_const
- Save per-particle invariants to CSV with suffixes: *_virial, *_strain.
- Merge aggregated series into global table with suffixes *_virial, *_strain.
- Generate plots for both stress types and label titles accordingly.
- Still computes global engineering strains from wall motion and void-ratio merges.

Usage:
  Edit input_dir/output_dir at bottom and run:
      python postprocess_make_plots_standalone_extplots_v2_patched.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ helpers ------------------------
def apply_strain_convention(eps, convention="tension_positive"):
    if convention == "compression_positive":
        return -eps
    return eps

def find_first(df, names, required=False, default=None):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"Required column not found. Tried: {names}")
    return default
def detect_stress_convention_for_constitutive(df_pp):
    # Use the mean trace of S_const as a heuristic
    tr_mean = (df_pp["S11_const"] + df_pp["S22_const"] + df_pp["S33_const"]).mean()
    return "compression_positive" if tr_mean > 0 else "tension_positive"

def invariants_from_components(df, colmap, *, stress_convention="tension_positive"):
    """
    stress_convention: "tension_positive" (default) or "compression_positive".
    """
    import numpy as np

    sxx = df[colmap["sxx"]].to_numpy()
    syy = df[colmap["syy"]].to_numpy()
    szz = df[colmap["szz"]].to_numpy()
    sxy = df[colmap["sxy"]].to_numpy()
    sxz = df[colmap["sxz"]].to_numpy()
    syz = df[colmap["syz"]].to_numpy()

    n = len(df)
    sig = np.zeros((n,3,3), dtype=float)
    sig[:,0,0] = sxx; sig[:,1,1] = syy; sig[:,2,2] = szz
    sig[:,0,1] = sig[:,1,0] = sxy
    sig[:,0,2] = sig[:,2,0] = sxz
    sig[:,1,2] = sig[:,2,1] = syz

    tr = np.trace(sig, axis1=1, axis2=2)
    # p definition depends on stress sign convention
    if str(stress_convention).startswith("compression"):
        p = tr / 3.0                # compression-positive stresses → p = +tr/3
    else:
        p = -tr / 3.0               # tension-positive stresses → p = -tr/3

    eye = np.eye(3)
    s = sig - (tr/3.0)[:,None,None] * eye   # deviatoric is independent of convention
    J2 = 0.5*np.einsum("nij,nij->n", s, s)
    q  = np.sqrt(3.0*J2)
    vm = np.sqrt(1.5*np.einsum("nij,nij->n", s, s))

    w = np.linalg.eigvalsh(sig)     # ascending
    s1,s2,s3 = w[:,2], w[:,1], w[:,0]
    tau_oct = (1.0/3.0)*np.sqrt((s1-s2)**2 + (s2-s3)**2 + (s3-s1)**2)

    out = df.copy()
    out["p"] = p
    out["q"] = q
    out["von_mises"] = vm
    out["tau_oct"] = tau_oct
    out["s1"] = s1; out["s2"] = s2; out["s3"] = s3
    out["sig_diff"] = s1 - s3
    return out



def _wall_invariants_from_reaction(glob: pd.DataFrame, *, wall_convention: str = "compression"):
    """
    Compute invariants of *wall* stress per timestep.

    Source of wall stress:
      1) If columns [sx_reaction, sy_reaction, sz_reaction] exist, use them.
      2) Else, try to compute each axial stress from top-wall reaction force / area via _sigma_wall().
      3) Else, return None.

    Sign convention handling:
      - If wall_convention == "compression", we assume the provided wall stresses are compression-positive.
        We flip their sign before computing invariants so that p = -tr/3 yields compression-positive p.
      - If wall_convention == "tension", we use the values as given.
    """
    import numpy as _np
    need_any = any(c in glob.columns for c in ["sx_reaction","sy_reaction","sz_reaction"])
    # Try direct columns first
    if all(c in glob.columns for c in ["timestep","sx_reaction","sy_reaction","sz_reaction"]):
        sx = glob["sx_reaction"].to_numpy()
        sy = glob["sy_reaction"].to_numpy()
        sz = glob["sz_reaction"].to_numpy()
    else:
        # Attempt to construct from forces/area for each axis
        sx_series = _sigma_wall(glob, "x")
        sy_series = _sigma_wall(glob, "y")
        sz_series = _sigma_wall(glob, "z")
        # If all NaN, give up
        if sx_series.isna().all() and sy_series.isna().all() and sz_series.isna().all():
            return None
        sx = sx_series.to_numpy()
        sy = sy_series.to_numpy()
        sz = sz_series.to_numpy()

    flip = -1.0 if str(wall_convention).lower().startswith("comp") else 1.0
    out = []
    ts = glob["timestep"].to_numpy() if "timestep" in glob.columns else _np.arange(len(sx))
    for i in range(len(sx)):
        if _np.isnan(sx[i]) or _np.isnan(sy[i]) or _np.isnan(sz[i]):
            out.append({
                "timestep": float(ts[i]) if i < len(ts) else _np.nan,
                "p_wall": _np.nan, "q_wall": _np.nan, "von_mises_wall": _np.nan, "tau_oct_wall": _np.nan,
                "s1_wall": _np.nan, "s2_wall": _np.nan, "s3_wall": _np.nan, "sig_diff_wall": _np.nan
            })
            continue
        sig = _np.diag([flip*float(sx[i]), flip*float(sy[i]), flip*float(sz[i])])
        inv = _invariants_from_sigma(sig)
        out.append({
            "timestep": float(ts[i]) if i < len(ts) else _np.nan,
            "p_wall": inv["p"], "q_wall": inv["q"], "von_mises_wall": inv["von_mises"],
            "tau_oct_wall": inv["tau_oct"], "s1_wall": inv["s1"], "s2_wall": inv["s2"],
            "s3_wall": inv["s3"], "sig_diff_wall": inv["sig_diff"],
        })
    return pd.DataFrame(out)

# ----------------------------------------------------------------
# Wall reactions -> sigma_wall_axial (if available; quiet if missing)
# ----------------------------------------------------------------

def _invariants_from_sigma(sig):
    ''' 
    Compute basic invariants for one 3x3 tensor.
    Returns dict: p (compression+), q, von_mises, tau_oct, s1,s2,s3, sig_diff.
    ''' 
    import numpy as _np
    sig = _np.asarray(sig, dtype=float).reshape(3,3)
    tr = _np.trace(sig)
    p = -tr/3.0
    eye = _np.eye(3)
    s = sig + p*eye
    J2 = 0.5*_np.einsum("ij,ij->", s, s)
    q = _np.sqrt(3.0*J2)
    von_mises = _np.sqrt(1.5*_np.einsum("ij,ij->", s, s))
    w = _np.linalg.eigvalsh(sig)
    s1, s2, s3 = float(w[2]), float(w[1]), float(w[0])
    tau_oct = (1.0/3.0)*_np.sqrt((s1-s2)**2 + (s2-s3)**2 + (s3-s1)**2)
    return {"p": p, "q": q, "von_mises": von_mises, "tau_oct": tau_oct,
            "s1": s1, "s2": s2, "s3": s3, "sig_diff": s1 - s3}




def _sigma_wall(glob: pd.DataFrame, axial: str):
    """
    Return axial wall *stress* series for the chosen axis.
    Priority:
      1) Direct stress columns in global_summary (sx_reaction/sy_reaction/sz_reaction, or sigma_**)
      2) Compute from a top-wall reaction *force* / area if found
      3) NaN
    """
    axial = (axial or "z").lower()
    if axial not in ("x", "y", "z"):
        axial = "z"

    # 1) Direct stress columns (your schema)
    direct_map = {
        "x": ["sx_reaction", "sigma_xx_wall", "sigma_xx_reaction"],
        "y": ["sy_reaction", "sigma_yy_wall", "sigma_yy_reaction"],
        "z": ["sz_reaction", "sigma_zz_wall", "sigma_zz_reaction"],
    }
    for cname in direct_map[axial]:
        if cname in glob.columns:
            return glob[cname]

    # 2) Compute from reaction *force* / area if a force column exists
    force_cols = [
        f"F_top_{axial}", f"F_{axial}_top", f"F{axial}_top",
        f"reaction_top_{axial}", f"R_top_{axial}", f"force_top_{axial}"
    ]
    found_force = next((c for c in force_cols if c in glob.columns), None)
    if found_force is None:
        return pd.Series(np.nan, index=glob.index)

    # Area = product of spans orthogonal to axial
    ortho = dict(x=("y","z"), y=("x","z"), z=("x","y"))[axial]
    need = {f"{ortho[0]}_min", f"{ortho[0]}_max", f"{ortho[1]}_min", f"{ortho[1]}_max"}
    if not need <= set(glob.columns):
        return pd.Series(np.nan, index=glob.index)

    L1 = glob[f"{ortho[0]}_max"] - glob[f"{ortho[0]}_min"]
    L2 = glob[f"{ortho[1]}_max"] - glob[f"{ortho[1]}_min"]
    A  = (L1 * L2).replace(0.0, np.nan)
    return glob[found_force] / A








def weighted_mean(series, weights):
    if weights is None:
        return float(series.mean())
    w = np.asarray(weights); s = np.asarray(series)
    denom = w.sum()
    return float((s*w).sum()/denom) if denom != 0 else np.nan

def savefig(figs_dir, basename, save_png=True, save_pdf=True, dpi=160):
    if save_png:
        plt.savefig(figs_dir / f"{basename}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        plt.savefig(figs_dir / f"{basename}.pdf", bbox_inches="tight")
    plt.close()

def plot_xy(x, y, xlabel, ylabel, figs_dir, basename, *, title_suffix="", markersize=3,
            save_png=True, save_pdf=True, dpi=160):
    # Pretty labels
    param_explanations = {
        "p": r"$p$: mean stress (compression +)",
        "p'": r"$p'$: effective mean stress",
        "q": r"$q$: deviatoric stress = $\sqrt{3J_2}$",
        "epsilon_v": r"$\epsilon_v$: volumetric strain",
        "ε_v": r"$\epsilon_v$: volumetric strain",
        "epsilon_q": r"$\epsilon_q$: deviatoric strain",
        "ε_q": r"$\epsilon_q$: deviatoric strain",
        "τ_oct": r"$\tau_{oct}$: octahedral shear stress",
        "sigma_diff": r"$\sigma_1 - \sigma_3$: principal stress difference",
        "σ1 - σ3": r"$\sigma_1 - \sigma_3$: principal stress difference",
        "e": r"$e$: void ratio",
        "e / e0": r"$e/e_0$: normalized void ratio",
        "p' / σ0": r"$p'/\sigma_0$: normalized effective stress",
        "ln p": r"$\ln p$: log mean stress"
    }
    plt.figure()
    plt.scatter(x, y, s=markersize**2)  # s is marker area in points^2
    xlabel_expl = param_explanations.get(xlabel, xlabel)
    ylabel_expl = param_explanations.get(ylabel, ylabel)
    plt.xlabel(xlabel_expl)
    plt.ylabel(ylabel_expl)
    plt.grid(True, alpha=0.3)
    # Title mentions which stress flavor
    if title_suffix:
        plt.title(f"Comparison: {ylabel} vs {xlabel} [{title_suffix}]")
    else:
        plt.title(f"Comparison: {ylabel} vs {xlabel}")
    savefig(figs_dir, basename, save_png, save_pdf, dpi)

# ------------------------ main ------------------------
def main(input_dir, output_dir, *, axial="z", strain_flavor="linear",
         save_png=True, save_pdf=True, dpi=160):
    from pathlib import Path
    import numpy as np
    import pandas as pd

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # ---- validate strain_flavor ----
    strain_flavor = (strain_flavor or "linear").lower()
    if strain_flavor not in ("linear", "stvk"):
        raise ValueError(f"strain_flavor must be 'linear' or 'stvk', got {strain_flavor!r}")

    # ---- required files ----
    required = ["global_summary.csv"]
    missing = [f for f in required if not (input_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing in input_dir={input_dir}: {missing}")

    derived_dir = output_dir / "derived"
    figs_dir = output_dir / "figs"
    derived_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Global (wall) strains ----------------
    g = pd.read_csv(input_dir / "global_summary.csv")
    

    g["ex"] = apply_strain_convention(g["ex"], convention="compression_positive")
    g["ey"] = apply_strain_convention(g["ey"], convention="compression_positive")
    g["ez"] = apply_strain_convention(g["ez"], convention="compression_positive")
    #
    # volumetric strain (small strain)
    if all(c in g.columns for c in ["ex", "ey", "ez"]):
        g["epsilon_v"] = g["ex"] + g["ey"] + g["ez"]
    elif "epsilon_v" not in g.columns:
        g["epsilon_v"] = np.nan

    # Axial mapping for deviatoric strain
    axial = axial.lower()
    if axial not in ("x", "y", "z"):
        axial = "z"
    if axial == "z":
        axial_col, rad_cols = "ez", ("ex", "ey")
    elif axial == "y":
        axial_col, rad_cols = "ey", ("ex", "ez")
    else:
        axial_col, rad_cols = "ex", ("ey", "ez")
    for c in (axial_col, *rad_cols):
        if c not in g.columns:
            g["eps_q"] = np.nan
            break
    else:
        g["eps_q"] = (2.0/3.0) * (g[axial_col] - (g[rad_cols[0]] + g[rad_cols[1]])/2.0)

    # -------------- Void-ratio merge (optional) --------------
    vrs_path = input_dir / "void_ratio_summary.csv"
    if vrs_path.exists():
        vrs = pd.read_csv(vrs_path)
        if "timestep" in vrs.columns and "timestep" in g.columns:
            keep = [c for c in ["timestep", "container_volume", "solid_volume",
                                "void_ratio_e", "porosity_n"] if c in vrs.columns]
            g = g.merge(vrs[keep], on="timestep", how="left")

    # Normalize e/n if only one exists
    if "void_ratio_e" in g.columns and "porosity_n" not in g.columns:
        g["porosity_n"] = g["void_ratio_e"] / (1.0 + g["void_ratio_e"])
    if "porosity_n" in g.columns and "void_ratio_e" not in g.columns:
        g["void_ratio_e"] = g["porosity_n"] / (1.0 - g["porosity_n"]).replace(0, np.nan)

    # -------------- Per-particle stress invariants --------------
    # Choose file for strain-based (‘_const’) columns; virial columns are in the same file.
    primary_name = f"per_particle_stress_{'linear' if strain_flavor=='linear' else 'stvk'}.csv"
    candidates = [
        input_dir / primary_name,
        input_dir / "per_particle_stress.csv",
        # opposite flavor as a fallback
        input_dir / ("per_particle_stress_stvk.csv" if strain_flavor == "linear" else "per_particle_stress_linear.csv"),
    ]
    per_particle_path = next((p for p in candidates if p.exists()), None)

    df_inv_virial = df_inv_strain = None
    if per_particle_path is None:
        print("WARNING: No per-particle stress file found; skipping invariants.")
    else:
        df_pp = pd.read_csv(per_particle_path)

        # Column maps for both flavors
        virial_map = {
            "sxx": find_first(df_pp, ["S11_virial"], required=True),
            "syy": find_first(df_pp, ["S22_virial"], required=True),
            "szz": find_first(df_pp, ["S33_virial"], required=True),
            "sxy": find_first(df_pp, ["S12_virial"], required=True),
            "sxz": find_first(df_pp, ["S13_virial"], required=True),
            "syz": find_first(df_pp, ["S23_virial"], required=True),
        }
        strain_map = {
            "sxx": find_first(df_pp, ["S11_const"], required=True),
            "syy": find_first(df_pp, ["S22_const"], required=True),
            "szz": find_first(df_pp, ["S33_const"], required=True),
            "sxy": find_first(df_pp, ["S12_const"], required=True),
            "sxz": find_first(df_pp, ["S13_const"], required=True),
            "syz": find_first(df_pp, ["S23_const"], required=True),
        }

        # Per-particle invariants
        df_inv_virial = invariants_from_components(df_pp, virial_map).rename(columns={
            "p": "p_virial", "q": "q_virial", "von_mises": "von_mises_virial",
            "tau_oct": "tau_oct_virial", "s1": "s1_virial", "s2": "s2_virial",
            "s3": "s3_virial", "sig_diff": "sig_diff_virial"
        })
        const_conv = detect_stress_convention_for_constitutive(df_pp)
        df_inv_strain = invariants_from_components(
            df_pp,
            {
            "sxx": "S11_const", "syy": "S22_const", "szz": "S33_const",
            "sxy": "S12_const", "sxz": "S13_const", "syz": "S23_const",
            },
            stress_convention=const_conv,
        ).rename(columns={  # same renames you already do
            "p":"p_strain", "q":"q_strain", "von_mises":"von_mises_strain",
            "tau_oct":"tau_oct_strain","s1":"s1_strain","s2":"s2_strain",
            "s3":"s3_strain","sig_diff":"sig_diff_strain"
        })

        df_wall_inv = None  # will compute after global merge
        # Save per-particle invariants (audit)
        (derived_dir / "particle_invariants_virial.csv").write_text(df_inv_virial.to_csv(index=False))
        (derived_dir / f"particle_invariants_strain_{strain_flavor}.csv").write_text(df_inv_strain.to_csv(index=False))

        # Volume weights
        vol_col = find_first(df_pp, ["volume", "vol", "cell_vol", "Vcell"], required=False)
        weights = df_pp[vol_col].to_numpy() if (vol_col is not None and vol_col in df_pp.columns) else None

        # Aggregate by timestep (safe concat to avoid duplicate 'timestep')
        if "timestep" in df_pp.columns:
            inv_cols_vir = [c for c in df_inv_virial.columns if c.endswith("_virial")]
            inv_cols_str = [c for c in df_inv_strain.columns if c.endswith("_strain")]

            df_vir = pd.concat([df_pp[["timestep"]].reset_index(drop=True),
                                df_inv_virial[inv_cols_vir].reset_index(drop=True)], axis=1)
            df_str = pd.concat([df_pp[["timestep"]].reset_index(drop=True),
                                df_inv_strain[inv_cols_str].reset_index(drop=True)], axis=1)

            if weights is not None:
                df_vir["__w__"] = weights
                df_str["__w__"] = weights

            def _agg_block(df_block, suffix):
                def _w(d):
                    return d["__w__"].to_numpy() if "__w__" in d.columns else None
                grouped = df_block.groupby("timestep", group_keys=False)
                out = grouped.apply(lambda d: pd.Series({
                    f"p{suffix}":        weighted_mean(d[f"p{suffix}"],        _w(d)),
                    f"q{suffix}":        weighted_mean(d[f"q{suffix}"],        _w(d)),
                    f"tau_oct{suffix}":  weighted_mean(d[f"tau_oct{suffix}"],  _w(d)),
                    f"sig_diff{suffix}": weighted_mean(d[f"sig_diff{suffix}"], _w(d)),
                }))
                out.reset_index(inplace=True)
                return out

            agg_virial = _agg_block(df_vir, "_virial")
            agg_strain = _agg_block(df_str, "_strain")
        else:
            base_step = g["timestep"].iloc[0] if "timestep" in g.columns and len(g) > 0 else 0
            agg_virial = pd.DataFrame({
                "timestep": [base_step],
                "p_virial":        [weighted_mean(df_inv_virial["p_virial"],        weights)],
                "q_virial":        [weighted_mean(df_inv_virial["q_virial"],        weights)],
                "tau_oct_virial":  [weighted_mean(df_inv_virial["tau_oct_virial"],  weights)],
                "sig_diff_virial": [weighted_mean(df_inv_virial["sig_diff_virial"], weights)],
            })
            agg_strain = pd.DataFrame({
                "timestep": [base_step],
                "p_strain":        [weighted_mean(df_inv_strain["p_strain"],        weights)],
                "q_strain":        [weighted_mean(df_inv_strain["q_strain"],        weights)],
                "tau_oct_strain":  [weighted_mean(df_inv_strain["tau_oct_strain"],  weights)],
                "sig_diff_strain": [weighted_mean(df_inv_strain["sig_diff_strain"], weights)],
            })

        # Merge aggregated into g
        if "timestep" in g.columns:
            g = g.merge(agg_virial, on="timestep", how="left")
            g = g.merge(agg_strain, on="timestep", how="left")
        else:
            for k, v in agg_virial.iloc[0].items():
                if k != "timestep": g[k] = v
            for k, v in agg_strain.iloc[0].items():
                if k != "timestep": g[k] = v
    
    
    # -------------- Wall reaction invariants (from global summary) --------------
    df_wall_inv = _wall_invariants_from_reaction(g, wall_convention="compression")
    if df_wall_inv is not None:
        if "timestep" in g.columns and "timestep" in df_wall_inv.columns:
            g = g.merge(df_wall_inv, on="timestep", how="left")
        else:
            # No timestep? attach as standalone columns if single row
            for k, v in df_wall_inv.iloc[0].items():
                if k != "timestep":
                    g[k] = v
        # Save for audit
        (derived_dir / "wall_invariants.csv").write_text(df_wall_inv.to_csv(index=False))
#---------------------⚠️-------------------
    '''
    if "p_strain" in g.columns:
        if (g["p_strain"] < 0).all():
            g["p_strain"] = -g["p_strain"]
    '''
    # ---------------- Effective stress references ----------------
    # p' = p - u (if pore pressure u provided)
    if "u" in g.columns:
        if "p_virial" in g.columns: g["p_eff_virial"] = g["p_virial"] - g["u"]
        if "p_strain" in g.columns: g["p_eff_strain"] = g["p_strain"] - g["u"]
    else:
        if "p_virial" in g.columns: g["p_eff_virial"] = g["p_virial"]
        if "p_strain" in g.columns: g["p_eff_strain"] = g["p_strain"]

    # Normalization reference (prefer virial if available)
    ref_col = "p_eff_virial" if "p_eff_virial" in g.columns and g["p_eff_virial"].notna().any() else \
              ("p_eff_strain" if "p_eff_strain" in g.columns and g["p_eff_strain"].notna().any() else None)
    sigma0 = float(g[ref_col].dropna().iloc[0]) if ref_col else 1.0
    e0 = float(g["void_ratio_e"].dropna().iloc[0]) if "void_ratio_e" in g.columns and g["void_ratio_e"].notna().any() else 1.0
    g["peff_over_sigma0"] = (g[ref_col] / (sigma0 if sigma0 != 0 else 1.0)) if ref_col else np.nan
    if "void_ratio_e" in g.columns:
        g["e_over_e0"] = g["void_ratio_e"] / (e0 if e0 != 0 else 1.0)

    # Save augmented global
    (derived_dir / "global_augmented.csv").write_text(g.to_csv(index=False))

    # ------------------- plots -------------------
    ms = 1  # marker size (points)
    strain_title = f"strain-based ({'linear' if strain_flavor=='linear' else 'StVK'})"

    def safe_plot(xcol, ycol, name, title_suffix):
        if xcol in g.columns and ycol in g.columns and g[xcol].notna().any() and g[ycol].notna().any():
            plot_xy(g[xcol], g[ycol], xcol, ycol, figs_dir, name, title_suffix=title_suffix,
                    markersize=ms, save_png=save_png, save_pdf=save_pdf, dpi=dpi)

    # q–p
    safe_plot("p_virial", "q_virial", "qp_path_virial", "virial")
    safe_plot("p_strain", "q_strain", "qp_path_strain", strain_title)

    safe_plot(axial_col, "p_virial", "p_vs_axial_virial", "virial")
    safe_plot(axial_col, "q_virial", "q_vs_axial_virial", "virial")
    safe_plot("ez","p_virial", "p_path_virial", "virial")
    safe_plot("ez","p_virial",  "q_path_virial", "virial")
    safe_plot("timestep","p_virial", "p_path_virial", "virial")
    safe_plot("timestep","q_virial",  "q_path_virial", "virial")
    
    safe_plot("p_virial", "void_ratio_e", "void_ratio_ep_path_virial", "virial")
    safe_plot("q_virial", "void_ratio_e", "void_ratio_eq_path_virial", "virial")
    # εv vs p
    if "epsilon_v" in g.columns:
        safe_plot("p_virial", "epsilon_v", "ev_vs_p_virial", "virial")
        safe_plot("p_strain", "epsilon_v", "ev_vs_p_strain", strain_title)

    # τ_oct vs axial engineering strain
    if "tau_oct_virial" in g.columns:
        safe_plot(axial_col, "tau_oct_virial", "tau_oct_vs_axial_strain_virial", "virial")
    if "tau_oct_strain" in g.columns:
        safe_plot(axial_col, "tau_oct_strain", "tau_oct_vs_axial_strain_strain", strain_title)

    # e vs ln p
    if "void_ratio_e" in g.columns:
        if "p_virial" in g.columns and g["p_virial"].notna().any():
            x = np.log(np.clip(g["p_virial"].to_numpy(dtype=float), a_min=np.finfo(float).eps, a_max=None))
            plot_xy(x, g["void_ratio_e"], "ln p", "e", figs_dir, "e_vs_logp_virial",
                    title_suffix="virial", markersize=ms, save_png=save_png, save_pdf=save_pdf, dpi=dpi)
        if "p_strain" in g.columns and g["p_strain"].notna().any():
            x = np.log(np.clip(g["p_strain"].to_numpy(dtype=float), a_min=np.finfo(float).eps, a_max=None))
            plot_xy(x, g["void_ratio_e"], "ln p", "e", figs_dir, "e_vs_logp_strain",
                    title_suffix=strain_title, markersize=ms, save_png=save_png, save_pdf=save_pdf, dpi=dpi)

    # normalized e/e0 vs p'/σ0
    if "e_over_e0" in g.columns and "peff_over_sigma0" in g.columns and g["peff_over_sigma0"].notna().any():
        ref_name = "virial" if ref_col == "p_eff_virial" else ("strain-based" if ref_col == "p_eff_strain" else "")
        plot_xy(g["peff_over_sigma0"], g["e_over_e0"], "p' / σ0", "e / e0", figs_dir,
                "peff_over_sigma0_vs_e_over_e0", title_suffix=ref_name,
                markersize=ms, save_png=save_png, save_pdf=save_pdf, dpi=dpi)

    # Δσ vs p
    safe_plot("p_virial", "sig_diff_virial", "p_vs_sig_diff_virial", "virial")
    safe_plot("p_strain", "sig_diff_strain", "p_vs_sig_diff_strain", strain_title)

    # ε_q relations
    if "eps_q" in g.columns:
        if "q_virial" in g.columns:  safe_plot("eps_q", "q_virial", "eps_q_vs_q_virial", "virial")
        if "p_virial" in g.columns:  safe_plot("eps_q", "p_virial", "eps_q_vs_p_virial", "virial")
        if "q_strain" in g.columns:  safe_plot("eps_q", "q_strain", "eps_q_vs_q_strain", strain_title)
        if "p_strain" in g.columns:  safe_plot("eps_q", "p_strain", "eps_q_vs_p_strain", strain_title)
        if "epsilon_v" in g.columns:
            plot_xy(g["eps_q"], g["epsilon_v"], "ε_q", "epsilon_v", figs_dir, "eps_q_vs_eps_v",
                    title_suffix="", markersize=ms, save_png=save_png, save_pdf=save_pdf, dpi=dpi)
    if "p_wall" in g.columns:
        safe_plot("timestep","p_wall", "p_wall", "reaction")
        safe_plot("timestep","q_wall", "q_wall", "reaction")
        safe_plot("p_wall","void_ratio_e", "e-vs-p_wall", "reaction")
        safe_plot("p_wall","porosity_n", "n-vs-p_wall", "reaction")
        safe_plot("q_wall","porosity_n", "n-vs-q_wall", "reaction")
        safe_plot("q_wall","void_ratio_e", "e-vs-q_wall", "reaction")
        safe_plot("ez","q_wall", "ez-vs-q_wall", "reaction")
        safe_plot("ez","p_wall", "ez-vs-p_wall", "reaction")
        safe_plot("s3_wall", "porosity_n","n-vs-s3_wall", "reaction")
        safe_plot("s2_wall", "porosity_n","n-vs-s2wall", "reaction")
        safe_plot("s1_wall", "porosity_n","n-vs-s1_wall", "reaction")
    if 1:
        x = np.log(np.clip(g["p_wall"].to_numpy(dtype=float), a_min=np.finfo(float).eps, a_max=None))
        plot_xy(g["void_ratio_e"],x,"e", "ln p", figs_dir, "e_vs_logp_wall",
                    title_suffix="reaction", markersize=ms, save_png=save_png, save_pdf=save_pdf, dpi=dpi)
    if 1:
        x = np.log(np.clip(g["q_wall"].to_numpy(dtype=float), a_min=np.finfo(float).eps, a_max=None))
        plot_xy(g["void_ratio_e"],x,"e", "ln q", figs_dir, "e_vs_logq_wall",
                    title_suffix="reaction", markersize=ms, save_png=save_png, save_pdf=save_pdf, dpi=dpi)


    print(f"Wrote: {(derived_dir / 'global_augmented.csv')}")
    print(f"Figures in: {figs_dir}")
if __name__ == "__main__":

   # IMPORTANT: set paths
    #input_dir  = "/home/davood/Downloads/3d-paper-results/postprocessing/single_sphere/thin_N2"
    input_dir = "/home/davood/Downloads/3d-paper-results/postprocessing/single_sphere/thick-Aug28-1/csv"
    output_dir = "/home/davood/Downloads/3d-paper-results/postprocessing/single_sphere/thick-Aug28-1/plot4_dual"


    main(
        input_dir,
        output_dir,
        axial="z",
        strain_flavor="stvk",  # or "linear"
        save_png=True, save_pdf=False, dpi=160
    )

