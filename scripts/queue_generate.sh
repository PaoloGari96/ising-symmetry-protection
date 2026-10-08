#!/bin/sh
# Sequential Wolff data generation for the extension study (single core).
# Seeds are fixed and recorded in each output's metadata.
KC=0.44068679350977147
G="python3 scripts/generate_wolff.py"
$G --L 240 --K 0.203 0.078 --h 0 --N 4000 --skip 2 --seed 13 --out results/cfg_thesis_L240_h0_rep2.npz > results/cfg_thesis_L240_h0_rep2.log 2>&1
$G --L 256 --K $KC --h 0 --N 8000 --skip 2 --seed 21 --out results/cfg_nn_L256_h0.npz > results/cfg_nn_L256_h0.log 2>&1
s=30
for h in 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3; do
  s=$((s+1)); $G --L 120 --K $KC --h $h --N 3000 --skip 2 --seed $s --out results/cfg_nn_L120_h$h.npz > results/cfg_nn_L120_h$h.log 2>&1
done
$G --L 120 --K $KC --h 0 --N 3000 --skip 2 --seed 40 --out results/cfg_nn_L120_h0.npz > results/cfg_nn_L120_h0.log 2>&1
s=50
for h in 0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3; do
  s=$((s+1)); $G --L 60 --K $KC --h $h --N 3000 --skip 2 --seed $s --out results/cfg_nn_L60_h$h.npz > results/cfg_nn_L60_h$h.log 2>&1
done
$G --L 240 --K 0.203 0.078 --h 1e-4 --N 2000 --skip 2 --seed 61 --out results/cfg_thesis_L240_h1e-4.npz > results/cfg_thesis_L240_h1e-4.log 2>&1
$G --L 240 --K 0.203 0.078 --h 1e-3 --N 2000 --skip 2 --seed 62 --out results/cfg_thesis_L240_h1e-3.npz > results/cfg_thesis_L240_h1e-3.log 2>&1
$G --L 256 --K 0.2059 0.078 --h 0 --N 8000 --skip 2 --seed 71 --out results/cfg_k2c_L256_h0.npz > results/cfg_k2c_L256_h0.log 2>&1
python3 scripts/critical_line_extra.py > results/critical_line_K2_0.04.log 2>&1
echo done > results/queue_generate.done
