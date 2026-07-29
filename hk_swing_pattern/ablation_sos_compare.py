# -*- coding: utf-8 -*-
"""对比 SoS 新旧逻辑的 edge: 新(126日pos OR 均线纠缠) vs 旧(60日pos, 无纠缠)."""
import sys
from pathlib import Path
if sys.platform=="win32":
    try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    except: pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import backtest_entry as BE
from backtest_entry import compute_all_signals, build_conds, evaluate_entries
import ablation_sos as ABL

ASOF="2026-07-28"
print(f"loading HK data asof={ASOF} ...",flush=True)
pool, bag, bench = ABL.load_market("hk", ASOF)
print(f"pool={len(pool)} with_data={len(bag)}",flush=True)

orig_sos = BE.sos_per_day  # new logic (126d + entangle OR)

def run(label, sos_fn):
    BE.sos_per_day = sos_fn
    df = compute_all_signals(pool, bag, bench)
    if df is None:
        print(f"{label}: 无信号"); return
    n_sos = int(df["sos"].sum())
    print(f"\n{'='*72}\n{label}  (SoS触发样本={n_sos})\n{'='*72}")
    for mode in ["strong","weak"]:
        d = build_conds(df, mode, screen="bottom")
        print(f"\n--- mode={mode} ---")
        for name, fn in ABL.VARIANTS:
            mask = fn(d)
            evaluate_entries(d, mask, direction="long", label=f"{label} {mode} {name}", verbose=False)

# NEW
run("NEW_126d_OR_entangle", orig_sos)
# OLD: 60日pos, 纠缠门禁用(thresh<0)
run("OLD_60d_no_entangle", lambda c,o,h,l,v: orig_sos(c,o,h,l,v, pos_window=60, entangle_thresh=-1))
