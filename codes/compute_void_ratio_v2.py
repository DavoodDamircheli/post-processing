#!/usr/bin/env python3
import os
import re
import sys
import glob
import h5py
import numpy as np
# --- add near the imports ---
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

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

def list_timesteps(data_dir):
    tc_files = sorted(glob.glob(os.path.join(data_dir, "tc_*.h5")))
    steps = []
    for f in tc_files:
        m = re.search(r"tc_(\d+)\.h5$", os.path.basename(f))
        if m:
            steps.append((int(m.group(1)), f))
    return steps
def read_container_volume(wall_file):
    with h5py.File(wall_file, "r") as w:
        if "wall_info" not in w:
            raise KeyError(f"{wall_file} missing dataset /wall_info")
        ss = np.array(w["wall_info"])[:, 0] if w["wall_info"].ndim == 2 else np.array(w["wall_info"]).ravel()
        if ss.size != 6:
            raise ValueError(f"/wall_info expected 6 values but got shape {ss.shape} in {wall_file}")
        
        x_min, y_min, z_min, x_max, y_max, z_max = ss.tolist()
        
        # Shift so the minimum corner is at 0
        shift_x, shift_y, shift_z = -x_min, -y_min, -z_min
        x_min += shift_x; x_max += shift_x
        y_min += shift_y; y_max += shift_y
        z_min += shift_z; z_max += shift_z

        V_total = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        return V_total, (x_min, y_min, z_min, x_max, y_max, z_max)

def v0_read_container_volume(wall_file):
    with h5py.File(wall_file, "r") as w:
        if "wall_info" not in w:
            raise KeyError(f"{wall_file} missing dataset /wall_info")
        ss = np.array(w["wall_info"])[:, 0] if w["wall_info"].ndim == 2 else np.array(w["wall_info"]).ravel()
        if ss.size != 6:
            raise ValueError(f"/wall_info expected 6 values but got shape {ss.shape} in {wall_file}")
        x_min, y_min, z_min, x_max, y_max, z_max = ss.tolist()
        V_total = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        return V_total, (x_min, y_min, z_min, x_max, y_max, z_max)

def groups_like_particles(h5file):
    for name in h5file.keys():
        if re.match(r"P_\d+$", name):
            yield name

def particle_volume_from_dataset(g):
    if "volume" in g:
        v = np.array(g["volume"])
        return float(v.squeeze())
    return None

def particle_volume_from_hull(g):
    if "CurrPos" not in g:
        return None
    P = np.array(g["CurrPos"])
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 4:
        return 0.0
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(P)
        return float(hull.volume)
    except Exception:
        return 0.0

def particle_volume_from_shell(g, global_outer, global_inner):
    outer_keys = ["r_outer", "radius_outer", "outer_radius", "R_outer", "Rout"]
    inner_keys = ["r_inner", "radius_inner", "inner_radius", "R_inner", "Rin"]
    R_outer = next((float(np.array(g[k]).squeeze()) for k in outer_keys if k in g), global_outer)
    R_inner = next((float(np.array(g[k]).squeeze()) for k in inner_keys if k in g), global_inner)
    return (4.0/3.0) * np.pi * (R_outer**3 - R_inner**3)

def compute_for_timestep(tc_file, wall_file, volume_method="auto", strict=False, shell_r_outer=None, shell_r_inner=None):
    V_total, bbox = read_container_volume(wall_file)
    V_solid = 0.0
    num_particles = 0
    used = {"dataset": 0, "hull": 0, "shell": 0}
    missing = 0
    per_particle = []

    with h5py.File(tc_file, "r") as f:
        for pname in groups_like_particles(f):
            g = f[pname]
            num_particles += 1
            v = None
            method_used = None

            if volume_method in ("dataset", "auto"):
                v = particle_volume_from_dataset(g)
                if v is not None:
                    method_used = "dataset"

            if v is None and volume_method in ("hull", "auto"):
                v = particle_volume_from_hull(g)
                method_used = "hull"

            if v is None and volume_method == "shell":
                v = particle_volume_from_shell(g, shell_r_outer, shell_r_inner)
                method_used = "shell"

            if v is None:
                missing += 1
                if strict:
                    raise ValueError(f"Missing volume for {pname} in {tc_file}")
                continue

            V_solid += v
            used[method_used] += 1
            per_particle.append({
                "timestep_file": os.path.basename(tc_file),
                "particle": pname,
                "volume": v,
                "method": method_used
            })

    e = (V_total - V_solid) / V_solid if V_solid > 0 else np.nan
    n = 1.0 - V_solid / V_total if V_solid > 0 else np.nan

    summary = {
        "timestep_file": os.path.basename(tc_file),
        "wall_file": os.path.basename(wall_file),
        "x_min": bbox[0], "y_min": bbox[1], "z_min": bbox[2],
        "x_max": bbox[3], "y_max": bbox[4], "z_max": bbox[5],
        "container_volume": V_total,
        "solid_volume": V_solid,
        "void_ratio_e": e,
        "porosity_n": n,
        "num_particles": num_particles,
        "num_missing_volumes": missing,
        "used_dataset": used["dataset"],
        "used_hull": used["hull"],
        "used_shell": used["shell"]
    }
    return summary, per_particle

#-----------------------------------------------------------------V

def main(data_dir, out_csv, per_particle_csv, volume_method="auto", strict=False, shell_r_outer=None, shell_r_inner=None):
    summaries = []
    per_particle_all = []

    any_steps = False
    for t, tc_file, wall_file in iter_valid_steps(data_dir):
        any_steps = True
        print(f"Opening: t={t}  tc='{tc_file}'  wall='{wall_file}'")
        summary, per_particle = compute_for_timestep(
            tc_file, wall_file, volume_method, strict, shell_r_outer, shell_r_inner
        )
        summary["timestep"] = t
        summaries.append(summary)
        per_particle_all.extend(per_particle)

    # If nothing valid was found, fail gracefully with a clear message
    if not any_steps:
        raise RuntimeError(f"No valid (tc_*.h5, wall_*.h5) pairs found under {data_dir}")

    if not summaries:
        raise RuntimeError("All timesteps were skipped; no summary rows to write. See warnings above.")

    # Write CSVs safely even if per-particle is empty
    import csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    if per_particle_all:
        os.makedirs(os.path.dirname(per_particle_csv), exist_ok=True)
        with open(per_particle_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_particle_all[0].keys()))
            w.writeheader()
            w.writerows(per_particle_all)
    else:
        print("NOTE: no per-particle rows written (empty per_particle_all).")



#-----------------------------------------------------------------V
if __name__ == "__main__":




    #-----------aggregate---------------
    #data_dir="/media/davood/093c4011-b7d0-4917-a86a-7c2fb7e4c748/project_data/force-trace/3d-ordinary-shpes/new_setup_hollow_sphere/N400"
    #data_dir = "/media/davood/093c4011-b7d0-4917-a86a-7c2fb7e4c748/project_data/result-periDEM/compress/single-sphere/N2"
    #output_dir = "/home/davood/Downloads/3d-paper-results/postprocessing/multi_sphere_compress/R07_N400_aug24_1" 
    
    #---------single-------------
    #data_dir = "/home/davood/Downloads/3d-paper-results/single_sphere/master-thin-N2"
    data_dir="/home/davood/projects/beta_perigrain_v2/examples_output/compress/single_sphere/N1R34/" 
    output_dir="/home/davood/Downloads/3d-paper-results/postprocessing/single_sphere/thin-Aug28-1/csv-1" 
    #output_dir = "/home/davood/Downloads/3d-paper-results/postprosessing/single_sphere/thin_N2"
 
    os.makedirs(output_dir, exist_ok=True)

    out_csv = os.path.join(output_dir, "void_ratio_summary.csv")
    per_particle_csv = os.path.join(output_dir, "per_particle_volumes.csv")

    volume_method = "shell"  # "auto", "dataset", "hull", or "shell"
    #------------------------- SOS-------------------------------
    #---- Dont forget to modify the radios and the shel thickness
    #------------------------------------------------------------

    radi = 200e-3 
    #radi = 1e-3 
    shell_r_outer = radi 
    shell_r_inner = (3/4)*shell_r_outer  
    #shell_r_inner = 0.7*shell_r_outer  

    main(data_dir, out_csv, per_particle_csv, volume_method, False, shell_r_outer, shell_r_inner)

