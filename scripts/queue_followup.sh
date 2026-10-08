#!/bin/bash
# Follow-up: null test at the critical point of the thesis model (K1 = 0.2059, K2 = 0.078, L = 256).
cd "$(dirname "$0")/.."
while [ ! -f results/queue_robustness.done ]; do sleep 20; done
python scripts/analyse_protection.py results/cfg_k2c_L256_h0.npz results/prot_k2c_L256_h0.npz \
       --bs 4 8 --bases thesis_theta phi > results/prot_k2c_L256_h0.log 2>&1
echo done > results/queue_followup.done
