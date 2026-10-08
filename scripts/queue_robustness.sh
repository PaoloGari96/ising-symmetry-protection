#!/bin/bash
# Robustness checks for report section 10, run sequentially (2-core machine).
cd "$(dirname "$0")/.."
python scripts/autocorrelation.py            > results/autocorr.log 2>&1
python scripts/leakage_controls.py           > results/leakage_controls.log 2>&1
python scripts/critical_line_refine.py       > results/critical_line_K2_0.04_refined.log 2>&1
python scripts/error_calibration.py          > results/error_calibration.log 2>&1
python scripts/kw_protection_scan.py --Nmax 18 --out results/kw_scan_N18.npz > results/kw_scan_N18.log 2>&1
echo done > results/queue_robustness.done
