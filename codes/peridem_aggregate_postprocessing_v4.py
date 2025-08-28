#!/usr/bin/env python3
"""
PeriDEM Aggregate Post‑Processing + Scatter Plots (separate virial/strain)
=======================================================================

Builds an aggregate time series from CSVs and generates **scatter** plots.
Key guarantees:
  - Compute macro tensors first (volume‑weighted), then invariants.
  - Virial and strain‑based curves are never mixed on the same axes.
  - Optional auto‑flip of virial macro stress per timestep based on compression.

Inputs expected in `input_dir`:
  - global_summary.csv
  - per_particle_stress.csv  (contains BOTH virial and constitutive stresses)
  - per_particle_strain.csv  (optional; not required here)
  - void_ratio_summary.csv   (optional; or *_all.csv)

Outputs in `output_dir`:
  - aggregate_timeseries.csv
  - figs_scatter/*.png (or .pdf if enabled)

Usage (direct call is set in __main__ at bottom), or as a module:
    main(input_dir, output_dir,
         axial='z', strain_flavor='stvk',
         save_png=True, save_pdf=False, dpi=160,
         auto_flip_virial=True,
         extra_specs=None)

`extra_specs` is a list of dicts like:
    {"x": "e_axial", "y": "q_virial", "name": "q_vs_eaxial_virial", "title": "q vs e_axial [virial]"}
You can also use log‑x with "ln <column>", e.g. x="ln p_virial".
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Pretty axis label dictionary
# ----------------------------
LABELS: Dict[str, str] = {
    "e_axial": r"Axial engineering strain $e_\mathrm{axial}$ (compression $+$)",
    "epsilon_v": r"Volumetric strain $\epsilon_v$",
    "eps_q": r"Deviatoric strain $\epsilon_q$",
    "p": r"Mean stress $p$ (compression $+$)",
    "q": r"$q = \sqrt{3J_2}$",
    "von_mises": r"von Mises $\sigma_\mathrm{vM}$",
    "tau_oct": r"Octahedral shear $\tau_\mathrm{oct}$",
    "s1": r"$\sigma_1$ (major)",
    "s2": r"$\sigma_2$",
    "s3": r"$\sigma_3$ (minor)",
    "sig_diff": r"$\sigma_1-\sigma_3$",
    "void_ratio_e": r"Void ratio $e$",
    "porosity_n": r"Porosity $n$",
    "ln p": r"$\ln p$",
}

# -----------------------------
# Utility: filesystem + reading
# -----------------------------
def _cap_by_timestep(df: Optional[pd.DataFrame], n: Optional[int]) -> Optional[pd.DataFrame]:
    if df is None or n is None:
        return df
    if "timestep" not in df.columns:
        return df
    return df[df["timestep"] <= int(n)].copy()

def _read_csv_maybe(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            warnings.warn(f"Failed to read {path}: {e}")
            return None
    return None

# ------------------------------------------
# Axial mapping (for deviatoric wall strain)
# ------------------------------------------

def _axial_map(axial: str) -> Tuple[str, Tuple[str, str]]:
    a = (axial or "z").lower()
    if a not in ("x", "y", "z"):
        a = "z"
    if a == "z":
        return "ez", ("ex", "ey")
    if a == "y":
        return "ey", ("ex", "ez")
    return "ex", ("ey", "ez")

# ------------------------
# Tensor / invariant math
# ------------------------
def _wall_invariants_from_reaction(glob: pd.DataFrame, *, wall_convention: str = "tension") -> Optional[pd.DataFrame]:
    """
    Build a diagonal stress from wall reactions and compute invariants.
    wall_convention:
      - "tension": sx/sy/sz are tension-positive (standard); use as-is.
      - "compression": sx/sy/sz are compression-positive; negate before invariants.
    Returns [timestep, p_wall, q_wall, von_mises_wall, tau_oct_wall, s1_wall, s2_wall, s3_wall, sig_diff_wall].
    """
    need = {"timestep","sx_reaction","sy_reaction","sz_reaction"}
    if not need <= set(glob.columns):
        return None

    flip = -1.0 if str(wall_convention).lower().startswith("comp") else 1.0
    out = []
    for _, r in glob.iterrows():
        sx, sy, sz = r["sx_reaction"], r["sy_reaction"], r["sz_reaction"]
        if pd.isna(sx) or pd.isna(sy) or pd.isna(sz):
            out.append({k: np.nan for k in ["timestep","p_wall","q_wall","von_mises_wall","tau_oct_wall","s1_wall","s2_wall","s3_wall","sig_diff_wall"]} | {"timestep": r["timestep"]})
            continue
        sig = np.diag([flip*float(sx), flip*float(sy), flip*float(sz)])
        inv = _invariants_from_sigma(sig)  # uses p = -tr/3 (compression +)
        out.append({
            "timestep": r["timestep"],
            "p_wall": inv["p"], "q_wall": inv["q"], "von_mises_wall": inv["von_mises"],
            "tau_oct_wall": inv["tau_oct"], "s1_wall": inv["s1"], "s2_wall": inv["s2"],
            "s3_wall": inv["s3"], "sig_diff_wall": inv["sig_diff"],
        })
    return pd.DataFrame(out)

def _invariants_from_sigma(sig: np.ndarray) -> Dict[str, float]:
    """Compute p, q, von Mises, tau_oct, principal stresses from a 3x3 tensor.
    Conventions: p > 0 in compression, i.e. p = -trace(sigma)/3.
    """
    if sig.shape != (3, 3):
        raise ValueError("sigma must be 3x3")
    # Enforce symmetry (virial can be slightly non-symmetric)
    sig = 0.5 * (sig + sig.T)
    tr = np.trace(sig)
    p = -tr / 3.0
    I = np.eye(3)
    s = sig - (tr / 3.0) * I
    J2 = 0.5 * float(np.tensordot(s, s))
    q = float(np.sqrt(max(0.0, 3.0 * J2)))
    vM = float(np.sqrt(max(0.0, 1.5 * np.tensordot(s, s))))
    evals = np.linalg.eigvalsh(sig)
    s1, s2, s3 = float(evals[2]), float(evals[1]), float(evals[0])
    tau_oct = (1.0 / 3.0) * np.sqrt((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
    return {
        "p": p,
        "q": q,
        "von_mises": vM,
        "tau_oct": float(tau_oct),
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "sig_diff": s1 - s3,
    }

# -----------------------------------------------------
# Wall kinematics -> strains, area, specimen dimensions
# -----------------------------------------------------

def _dims_from_bounds(row: pd.Series) -> Tuple[float, float, float]:
    Lx = float(row["x_max"]) - float(row["x_min"]) if {"x_max","x_min"} <= set(row.index) else np.nan
    Ly = float(row["y_max"]) - float(row["y_min"]) if {"y_max","y_min"} <= set(row.index) else np.nan
    Lz = float(row["z_max"]) - float(row["z_min"]) if {"z_max","z_min"} <= set(row.index) else np.nan
    return Lx, Ly, Lz


def _wall_strains(glob: pd.DataFrame, axial: str) -> pd.DataFrame:
    required = {"x_min","x_max","y_min","y_max","z_min","z_max"}
    if not required.issubset(glob.columns):
        warnings.warn("global_summary.csv missing bounds; cannot compute wall strains.")
        return glob.assign(ex=np.nan, ey=np.nan, ez=np.nan, epsilon_v=np.nan, eps_q=np.nan)

    if "timestep" in glob.columns:
        glob = glob.sort_values("timestep").reset_index(drop=True)

    Lx0, Ly0, Lz0 = _dims_from_bounds(glob.iloc[0])

    def _row_strains(row: pd.Series) -> Tuple[float, float, float]:
        Lx, Ly, Lz = _dims_from_bounds(row)
        ex = (Lx0 - Lx) / Lx0 if np.isfinite(Lx0) and Lx0 != 0 else np.nan
        ey = (Ly0 - Ly) / Ly0 if np.isfinite(Ly0) and Ly0 != 0 else np.nan
        ez = (Lz0 - Lz) / Lz0 if np.isfinite(Lz0) and Lz0 != 0 else np.nan
        return ex, ey, ez

    ex, ey, ez = zip(*glob.apply(_row_strains, axis=1))
    glob = glob.assign(ex=ex, ey=ey, ez=ez)

    axial_col, rad_cols = _axial_map(axial)
    eps_q = []
    for _, r in glob.iterrows():
        try:
            e_ax = float(r[axial_col])
            e_r1 = float(r[rad_cols[0]])
            e_r2 = float(r[rad_cols[1]])
            eps_q.append((2.0/3.0) * (e_ax - 0.5 * (e_r1 + e_r2)))
        except Exception:
            eps_q.append(np.nan)
    glob = glob.assign(epsilon_v=glob["ex"] + glob["ey"] + glob["ez"], eps_q=eps_q)
    return glob

# ----------------------------------------------------------------
# Wall reactions -> sigma_wall_axial (if available; quiet if missing)
# ----------------------------------------------------------------


def _sigma_wall(glob: pd.DataFrame, axial: str) -> pd.Series:
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


# -----------------------------------------------------------
# Volume-weighted macro stress from per-particle stress table
# -----------------------------------------------------------

def _macro_from_particles(pstress: pd.DataFrame, flavor: str) -> pd.DataFrame:
    """Compute volume-weighted macro sigma per timestep for a chosen stress flavor.

    flavor: 'virial' or one of {'const','linear','stvk'}; we map to columns suffixed accordingly.
    Returns a dataframe with columns ['timestep','S11','S22','S33','S12','S13','S23'] for that flavor.
    """
    if "timestep" not in pstress.columns:
        raise ValueError("per_particle_stress.csv must have a 'timestep' column")

    flavor = (flavor or "stvk").lower()
    suf = "virial" if flavor == "virial" else "const"

    needed = {f"S11_{suf}", f"S22_{suf}", f"S33_{suf}", f"S12_{suf}", f"S13_{suf}", f"S23_{suf}", "volume"}
    if not needed.issubset(pstress.columns):
        missing = needed - set(pstress.columns)
        raise ValueError(f"per_particle_stress missing columns: {missing}")

    rows = []
    for t, g in pstress.groupby("timestep"):
        V = g["volume"].astype(float)
        Vtot = V.sum()
        if Vtot == 0 or not np.isfinite(Vtot):
            rows.append({"timestep": t, "S11": np.nan, "S22": np.nan, "S33": np.nan, "S12": np.nan, "S13": np.nan, "S23": np.nan})
            continue
        w = V / Vtot
        S11 = np.sum(w * g[f"S11_{suf}"].astype(float))
        S22 = np.sum(w * g[f"S22_{suf}"].astype(float))
        S33 = np.sum(w * g[f"S33_{suf}"].astype(float))
        S12 = np.sum(w * g[f"S12_{suf}"].astype(float))
        S13 = np.sum(w * g[f"S13_{suf}"].astype(float))
        S23 = np.sum(w * g[f"S23_{suf}"].astype(float))
        rows.append({"timestep": t, "S11": S11, "S22": S22, "S33": S33, "S12": S12, "S13": S13, "S23": S23})
    return pd.DataFrame(rows)

# ---------------------------------------------
# Auto flip virial macro tensor if needed
# ---------------------------------------------

def _auto_flip_virial_macro(df_macro_vir: pd.DataFrame, df_glob: pd.DataFrame, *, axial: str, eps_thresh: float = 1e-7) -> pd.DataFrame:
    """Detect and fix sign polarity of virial macro stress per timestep.

    If the trace of the virial macro tensor is positive (tension) while the
    specimen is clearly in compression (e_axial > eps_thresh), we flip the
    entire tensor (multiply by -1). Returns copy with 'flip_virial' in {+1,-1}.
    """
    axial_col, _ = _axial_map(axial)
    if axial_col not in df_glob.columns:
        out = df_macro_vir.copy()
        out["flip_virial"] = 1
        return out
    g = df_glob[["timestep", axial_col]].rename(columns={axial_col: "e_axial_like"})
    m = df_macro_vir.merge(g, on="timestep", how="left")
    flips = []
    for _, r in m.iterrows():
        tr = float((r.get("S11", 0.0) or 0.0) + (r.get("S22", 0.0) or 0.0) + (r.get("S33", 0.0) or 0.0))
        eax = r.get("e_axial_like", np.nan)
        flip = 1
        try:
            if pd.notna(eax) and float(eax) > eps_thresh and tr > 0.0:
                flip = -1
        except Exception:
            flip = 1
        flips.append(flip)
    m["flip_virial"] = flips
    for c in ("S11","S22","S33","S12","S13","S23"):
        m[c] = m[c] * m["flip_virial"]
    return m[["timestep","S11","S22","S33","S12","S13","S23","flip_virial"]]

# ---------------------------------------
# Void ratio: read or compute if missing
# ---------------------------------------

def _parse_timestep_from_name(name: str) -> Optional[int]:
    m = re.search(r"(\d{3,})", str(name))
    return int(m.group(1)) if m else None


def _void_from_file(void_csv: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if void_csv is None:
        return None
    df = void_csv.copy()
    if "timestep" not in df.columns:
        if "timestep_file" in df.columns:
            df["timestep"] = df["timestep_file"].apply(_parse_timestep_from_name)
        else:
            warnings.warn("void_ratio_summary has no timestep nor timestep_file; skipping join.")
            return None
    keep = [c for c in ["timestep","container_volume","solid_volume","void_ratio_e","porosity_n"] if c in df.columns]
    return df[keep].drop_duplicates(subset=["timestep"]) if "timestep" in df.columns else None


def _void_compute_fallback(glob: pd.DataFrame, pstress: pd.DataFrame) -> pd.DataFrame:
    if not {"x_min","x_max","y_min","y_max","z_min","z_max"} <= set(glob.columns):
        warnings.warn("Cannot compute container volume from global_summary; void ratio fallback unavailable.")
        return pd.DataFrame()
    dims = glob.assign(
        Lx = glob["x_max"] - glob["x_min"],
        Ly = glob["y_max"] - glob["y_min"],
        Lz = glob["z_max"] - glob["z_min"],
    )
    contV = dims.assign(container_volume=dims["Lx"]*dims["Ly"]*dims["Lz"]).loc[:, ["timestep","container_volume"]]
    solidV = pstress.groupby("timestep")["volume"].sum().reset_index().rename(columns={"volume":"solid_volume"})
    out = contV.merge(solidV, on="timestep", how="inner")
    out["void_ratio_e"] = (out["container_volume"] - out["solid_volume"]) / out["solid_volume"].replace(0.0, np.nan)
    out["porosity_n"] = out["void_ratio_e"] / (1.0 + out["void_ratio_e"])
    return out

# -------------------
# Plotting utilities
# -------------------

def _savefig(fig: plt.Figure, path_base: Path, save_png: bool, save_pdf: bool, dpi: int):
    if save_png:
        fig.savefig(str(path_base.with_suffix('.png')), dpi=dpi, bbox_inches='tight')
    if save_pdf:
        fig.savefig(str(path_base.with_suffix('.pdf')), bbox_inches='tight')
    plt.close(fig)


def scatter_xy(x, y, out_path: Path, *, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               title: Optional[str] = None, markersize: int = 2, alpha: float = 0.9,
               save_png: bool = True, save_pdf: bool = False, dpi: int = 160) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(x, y, s=markersize**2, alpha=alpha)
    ax.set_xlabel(LABELS.get(xlabel, xlabel or ""))
    ax.set_ylabel(LABELS.get(ylabel, ylabel or ""))
    if title:
        ax.set_title(title)
    ax.grid(True, zorder=0, alpha=0.3)
    _savefig(fig, out_path, save_png, save_pdf, dpi)


def _series_from_expr(df: pd.DataFrame, name: str) -> np.ndarray:
    if name.startswith("ln "):
        col = name[3:]
        if col not in df.columns:
            return np.array([])
        vals = df[col].to_numpy(dtype=float)
        pos = vals > np.finfo(float).eps
        out = np.full_like(vals, np.nan)
        out[pos] = np.log(vals[pos])
        return out
    return df[name].to_numpy(dtype=float) if name in df.columns else np.array([])


def _build_default_specs(df: pd.DataFrame) -> List[Dict[str,str]]:
    specs: List[Dict[str,str]] = []
    def add_pair(x: str, y_base: str, name_base: str, title_base: str):
        yv = f"{y_base}_virial"; ys = f"{y_base}_strain"
        if yv in df.columns:
            specs.append({"x": x, "y": yv, "name": f"{name_base}_virial", "title": f"{title_base} [virial]"})
        if ys in df.columns:
            specs.append({"x": x, "y": ys, "name": f"{name_base}_strain", "title": f"{title_base} [strain-based]"})

    if "e_axial" in df.columns:
        specs.append({"x": "e_axial", "y": "p_virial", "name": "ez_vs_p_virial", "title": r"$ez$ vs $p$ [virial]"})
        specs.append({"x": "e_axial", "y": "q_virial", "name": "ez_vs_q_virial", "title": r"$ez$ vs $q$ [virial]"})
        add_pair("e_axial", "q", "q_vs_eaxial", r"$q$ vs $e_\mathrm{axial}$")
    if "epsilon_v" in df.columns:
        add_pair("epsilon_v", "p", "p_vs_epsv", r"$p$ vs $\epsilon_v$")
    if "p_virial" in df.columns or "p_strain" in df.columns:
        # Note: here x is the p_* column name itself, not the generic 'p'
        if "p_virial" in df.columns and "q_virial" in df.columns:
            specs.append({"x": "p_virial", "y": "q_virial", "name": "pq_path_virial", "title": r"Stress path $p$--$q$ [virial]"})
        if "p_strain" in df.columns and "q_strain" in df.columns:
            specs.append({"x": "p_strain", "y": "q_strain", "name": "pq_path_strain", "title": r"Stress path $p$--$q$ [strain-based]"})
    if "void_ratio_e" in df.columns:
        if "p_virial" in df.columns:
            specs.append({"x": "ln p_virial", "y": "void_ratio_e", "name": "e_vs_lnp_virial", "title": r"$e$ vs $\ln p$ [virial]"})
        if "p_virial" in df.columns:
                specs.append({"x": "p_virial", "y": "void_ratio_e", "name": "e_vs_p_virial", "title": r"$e$ vs $p$ [virial]"})
        if "p_strain" in df.columns:
            specs.append({"x": "ln p_strain", "y": "void_ratio_e", "name": "e_vs_lnp_strain", "title": r"$e$ vs $\ln p$ [strain-based]"})
        # in your _build_default_specs(df):
    if {"p_wall","q_wall"} <= set(df.columns):
        specs.append({"x": "timestep", "y": "p_wall", "name": "p_vs_time_reaction", "title": "p vs time [reaction]"})
        specs.append({"x": "timestep", "y": "q_wall", "name": "q_vs_time_reaction", "title": "q vs time [reaction]"})
        specs.append({"x":"p_wall", "y":"q_wall", "name":"pq_path_wall","title":"Stress path p–q [reaction]"})
        specs.append({"x": "e_axial", "y": "p_wall", "name": "ez_vs_p_wall", "title": r"$ez$ vs $p$ [reactiobn]"})
        specs.append({"x": "e_axial", "y": "q_wall", "name": "ez_vs_q_wall", "title": r"$ez$ vs $q$ [reactiobn]"})
    if "void_ratio_e" in df.columns and "p_wall" in df.columns:
         if "p_wall" in df.columns:
                specs.append({"x": "void_ratio_e", "y": "p_wall", "name": "e_vs_p_wall", "title": r"$e$ vs $p$ [reaction]"})
                specs.append({"x": "porosity_n", "y": "p_wall", "name": "n_vs_p_wall", "title": r"$n$ vs $p$ [reaction]"})
                specs.append({"x": "porosity_n", "y": "q_wall", "name": "n_vs_q_wall", "title": r"$n$ vs $p$ [reaction]"})
                specs.append({"x": "void_ratio_e", "y": "q_wall", "name": "e_vs_1_wall", "title": r"$e$ vs $q$ [reaction]"})
         specs.append({"x":"ln p_wall","y":"void_ratio_e","name":"e_vs_lnp_wall","title":"e vs ln p [wall]"})
    # and use add_pair(...) to include p_wall or q_wall vs strains if you like

    return specs

# ---------------------------
# Main pipeline
# ---------------------------

def main(input_dir: str | Path,
         output_dir: str | Path,
         axial: str = "z",
         strain_flavor: str = "stvk",
         save_png: bool = True,
         save_pdf: bool = False,
         dpi: int = 160,
         auto_flip_virial: bool = True,
         wall_convention: str = "compression",
         extra_specs: Optional[Iterable[Dict[str,str]]] = None,
         max_timestep: Optional[int] = None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    df_glob = _read_csv_maybe(input_dir/"global_summary.csv")
    df_pstress = _read_csv_maybe(input_dir/"per_particle_stress.csv")
    df_pstrain = _read_csv_maybe(input_dir/"per_particle_strain.csv")  # currently unused but kept

    if df_glob is None or df_pstress is None:
        raise FileNotFoundError("Need both global_summary.csv and per_particle_stress.csv in input_dir")
    if max_timestep is not None:
        print(f"[cap] Using timesteps <= {max_timestep}")
    # >>> NEW: cap early (affects everything downstream)
    df_glob    = _cap_by_timestep(df_glob,    max_timestep)
    df_pstress = _cap_by_timestep(df_pstress, max_timestep)
    if df_pstrain is not None:
        df_pstrain = _cap_by_timestep(df_pstrain, max_timestep)
    # Wall strains and sigma_wall
    df_glob = _wall_strains(df_glob, axial=axial)
    sigma_wall = _sigma_wall(df_glob, axial=axial)
    df_glob = df_glob.assign(sigma_wall_axial=sigma_wall)
    # Right after:
    # df_glob = _wall_strains(df_glob, axial=axial)
    # sigma_wall = _sigma_wall(df_glob, axial=axial)

    extras = {}
    if "sx_reaction" in df_glob.columns: extras["sigma_wall_xx"] = df_glob["sx_reaction"]
    if "sy_reaction" in df_glob.columns: extras["sigma_wall_yy"] = df_glob["sy_reaction"]
    if "sz_reaction" in df_glob.columns: extras["sigma_wall_zz"] = df_glob["sz_reaction"]

    df_glob = df_glob.assign(sigma_wall_axial=sigma_wall, **extras)
    df_wall_inv = _wall_invariants_from_reaction(df_glob, wall_convention=wall_convention)
    # Macro tensors from particles (virial and strain-based)
    df_macro_vir = _macro_from_particles(df_pstress, flavor="virial")
    if auto_flip_virial:
        df_macro_vir = _auto_flip_virial_macro(df_macro_vir, df_glob, axial=axial, eps_thresh=1e-7)
    else:
        df_macro_vir = df_macro_vir.assign(flip_virial=1)
    df_macro_vir = df_macro_vir.rename(columns={
        "S11":"S11_virial","S22":"S22_virial","S33":"S33_virial","S12":"S12_virial","S13":"S13_virial","S23":"S23_virial"
    })
    df_macro_str = _macro_from_particles(df_pstress, flavor=strain_flavor).rename(columns={
        "S11":"S11_strain","S22":"S22_strain","S33":"S33_strain","S12":"S12_strain","S13":"S13_strain","S23":"S23_strain"
    })

    # Join macros with global for same timesteps
    if "timestep" not in df_glob.columns:
        raise ValueError("global_summary.csv must include 'timestep' column")
    base = df_glob[["timestep","ex","ey","ez","epsilon_v","eps_q","x_min","x_max","y_min","y_max","z_min","z_max","sigma_wall_axial"]].copy()
    
    df_all = base.merge(df_macro_vir, on="timestep", how="left").merge(df_macro_str, on="timestep", how="left")
    

    if df_wall_inv is not None:
        df_all = df_all.merge(df_wall_inv, on="timestep", how="left")
    # Invariants from macro tensors
    def _inv_apply(row, prefix):
        S11, S22, S33 = row.get(f"S11_{prefix}"), row.get(f"S22_{prefix}"), row.get(f"S33_{prefix}")
        S12, S13, S23 = row.get(f"S12_{prefix}"), row.get(f"S13_{prefix}"), row.get(f"S23_{prefix}")
        if any(pd.isna([S11,S22,S33,S12,S13,S23])):
            return {"p": np.nan, "q": np.nan, "von_mises": np.nan, "tau_oct": np.nan, "s1": np.nan, "s2": np.nan, "s3": np.nan, "sig_diff": np.nan}
        sig = np.array([[S11, S12, S13],[S12, S22, S23],[S13, S23, S33]], dtype=float)
        return _invariants_from_sigma(sig)

    inv_v = df_all.apply(lambda r: _inv_apply(r, "virial"), axis=1, result_type='expand')
    inv_s = df_all.apply(lambda r: _inv_apply(r, "strain"), axis=1, result_type='expand')

    for k in inv_v.columns:
        df_all[f"{k}_virial"] = inv_v[k]
        df_all[f"{k}_strain"] = inv_s[k]

    # Axial engineering strain column (match chosen axial)
    axial_col, _ = _axial_map(axial)
    df_all = df_all.rename(columns={axial_col: "e_axial"})

    # Void ratio: prefer file, else compute
    void_csv = _read_csv_maybe(input_dir / "void_ratio_summary.csv")
    if void_csv is None:
        void_csv = _read_csv_maybe(input_dir / "void_ratio_summary_all.csv")
    df_void = _void_from_file(void_csv) if void_csv is not None else None
    if df_void is None:
         df_void = _void_compute_fallback(df_glob, df_pstress)
    else:
        df_void = _cap_by_timestep(df_void, max_timestep)
    
    #df_void = _void_compute_fallback(df_glob, df_pstress)

    #df_void = _cap_by_timestep(df_void, max_timestep)

    df_all = df_all.merge(df_void, on="timestep", how="left")

    # Select and order output columns
    cols = [
        "timestep", "e_axial", "epsilon_v", "eps_q",
        "void_ratio_e", "porosity_n",
        "sigma_wall_axial","sigma_wall_xx","sigma_wall_yy","sigma_wall_zz",
        "p_wall","q_wall","von_mises_wall","tau_oct_wall","s1_wall","s2_wall","s3_wall","sig_diff_wall",
        "flip_virial",
        "p_virial","q_virial","von_mises_virial","tau_oct_virial","s1_virial","s2_virial","s3_virial","sig_diff_virial",
        "p_strain","q_strain","von_mises_strain","tau_oct_strain","s1_strain","s2_strain","s3_strain","sig_diff_strain",
    ]
    cols = [c for c in cols if c in df_all.columns]
    out_ts = df_all[cols].sort_values("timestep")

    # Write aggregate time series
    out_csv = output_dir/"aggregate_timeseries.csv"
    out_ts.to_csv(out_csv, index=False)

    # ---- Scatter plots ----
    figs_dir = output_dir/"figs_scatter"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Built-in default specs
    specs = _build_default_specs(out_ts)
    if extra_specs:
        specs.extend(list(extra_specs))

    for spec in specs:
        x_name = spec["x"]; y_name = spec["y"]
        # Resolve data arrays (supports 'ln p_*')
        if not (x_name in out_ts.columns or x_name.startswith("ln ")):
            continue
        if y_name not in out_ts.columns:
            continue
        x = _series_from_expr(out_ts, x_name)
        y = _series_from_expr(out_ts, y_name)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        scatter_xy(x[mask], y[mask], figs_dir/spec["name"],
                   xlabel=x_name if x_name.startswith("ln ") else x_name,
                   ylabel=y_name,
                   title=spec.get("title"),
                   markersize=2, alpha=0.9,
                   save_png=save_png, save_pdf=save_pdf, dpi=dpi)

    print("Saved:", out_csv)
    print("Figures:", figs_dir)


if __name__ == "__main__":
    # --- Direct-call defaults (edit here) ---
    input_dir = "/home/davood/Downloads/3d-paper-results/postprocessing/multi_sphere_compress/R07_N400_aug24_1"
    output_dir = "/home/davood/Downloads/3d-paper-results/postprocessing/multi_sphere_compress/R07_N400_aug27_1/result_5"

    # Example: add your own scatter specs
    extra_specs = [
        {"x": "timestep", "y": "p_virial", "name": "p_vs_time_virial", "title": "p vs time [virial]"},
        {"x": "timestep", "y": "p_strain", "name": "p_vs_time_strain", "title": "p vs time [strain-based]"},
        {"x": "", "y": "p_strain", "name": "p_vs_time_strain", "title": "p vs time [strain-based]"},
    ]

    main(
        input_dir,
        output_dir,
        axial='z',
        strain_flavor='stvk',
        save_png=True,
        save_pdf=False,
        dpi=160,
        auto_flip_virial=True,
        wall_convention="compression",
        extra_specs=extra_specs,
        max_timestep=420,   # <<< cap here
        )

