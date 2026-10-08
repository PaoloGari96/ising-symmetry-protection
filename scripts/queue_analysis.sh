#!/bin/sh
# Sequential analyses (single core); waits for the data files produced by queue_generate.sh.
waitfor() { while [ ! -f "$1" ]; do sleep 20; done; sleep 5; }
P="python3 scripts"
waitfor results/cfg_thesis_L240_h0_rep2.npz
$P/analyse_protection.py results/cfg_thesis_L240_h0_rep2.npz results/prot_thesis_L240_h0_rep2.npz --bases thesis_theta phi > results/prot_thesis_L240_h0_rep2.log 2>&1
$P/analyse_protection.py results/cfg_nn_L240_h0.npz results/prot_nn_L240_h0.npz --bs 2 3 4 5 6 8 --bases thesis_theta phi phi_n8_plaq > results/prot_nn_L240_h0.log 2>&1
waitfor results/cfg_nn_L256_h0.npz
$P/run_mcrg.py results/cfg_nn_L256_h0.npz results/mcrg_nn_L256.npz --nlev 4 > results/mcrg_nn_L256.log 2>&1
waitfor results/cfg_nn_L120_h0.npz
$P/analyse_state_vs_field.py results/svf_nn_L120.npz --bs 2 3 4 5 6 --nbin 6 results/cfg_nn_L120_h0.npz results/cfg_nn_L120_h1e-5.npz results/cfg_nn_L120_h3e-5.npz results/cfg_nn_L120_h1e-4.npz results/cfg_nn_L120_h3e-4.npz results/cfg_nn_L120_h1e-3.npz results/cfg_nn_L120_h3e-3.npz > results/svf_nn_L120.log 2>&1
$P/analyse_field_response.py results/fr_nn_L120.npz results/cfg_nn_L120_h0.npz results/cfg_nn_L120_h1e-5.npz results/cfg_nn_L120_h3e-5.npz results/cfg_nn_L120_h1e-4.npz results/cfg_nn_L120_h3e-4.npz results/cfg_nn_L120_h1e-3.npz results/cfg_nn_L120_h3e-3.npz --bs 2 3 4 5 6 --bases phi > results/fr_nn_L120.log 2>&1
waitfor results/cfg_nn_L60_h3e-3.npz
# (b = 6 at L = 60 gives 10x10 blocked lattices with fully magnetized bins; the reported run uses b = 2-5)
$P/analyse_state_vs_field.py results/svf_nn_L60.npz --bs 2 3 4 5 --nbin 6 results/cfg_nn_L60_h0.npz results/cfg_nn_L60_h1e-5.npz results/cfg_nn_L60_h3e-5.npz results/cfg_nn_L60_h1e-4.npz results/cfg_nn_L60_h3e-4.npz results/cfg_nn_L60_h1e-3.npz results/cfg_nn_L60_h3e-3.npz > results/svf_nn_L60.log 2>&1
$P/analyse_field_response.py results/fr_nn_L60.npz results/cfg_nn_L60_h0.npz results/cfg_nn_L60_h1e-5.npz results/cfg_nn_L60_h3e-5.npz results/cfg_nn_L60_h1e-4.npz results/cfg_nn_L60_h3e-4.npz results/cfg_nn_L60_h1e-3.npz results/cfg_nn_L60_h3e-3.npz --bs 2 3 4 5 6 --bases phi > results/fr_nn_L60.log 2>&1
waitfor results/cfg_thesis_L240_h1e-3.npz
$P/analyse_state_vs_field.py results/svf_thesis_L240.npz --bs 2 3 4 5 6 8 --nbin 6 results/cfg_thesis_L240_h0.npz results/cfg_thesis_L240_h0_rep2.npz results/cfg_thesis_L240_h1e-4.npz results/cfg_thesis_L240_h1e-3.npz > results/svf_thesis_L240.log 2>&1
waitfor results/cfg_k2c_L256_h0.npz
$P/run_mcrg.py results/cfg_k2c_L256_h0.npz results/mcrg_k2c_L256.npz --nlev 4 > results/mcrg_k2c_L256.log 2>&1
echo done > results/queue_analysis.done
