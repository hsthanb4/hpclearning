import re
import os
import sys

new_files = [
    "diagnostics/lesson01_nsight_systems.qmd",
    "diagnostics/lesson02_nsight_compute.qmd",
    "diagnostics/lesson03_comm_deadlock.qmd",
    "diagnostics/lesson04_numerical_stability.qmd",
    "diagnostics/lesson05_invariant_audit.qmd",
    "cuda/lesson15_diffusion_policy.qmd",
    "cuda/lesson16_pointcloud_3d.qmd",
    "cuda/lesson17_qat_operators.qmd",
    "runtime/lesson09_gds_multimodal.qmd",
    "runtime/lesson10_ipc_shared_memory.qmd",
    "runtime/lesson11_dynamic_shapes.qmd",
    "inference/lesson09_tensorrt_plugins.qmd",
    "inference/lesson10_diffusion_engine.qmd",
    "inference/lesson11_edge_mixed_precision.qmd",
    "train/lesson15_multimodal_vla.qmd",
    "train/lesson16_microbatch_throughput.qmd",
    "rl/lesson13_continuous_action.qmd",
    "rl/lesson14_sim2real_flywheel.qmd",
]

total_tested = 0
failed = []

for fpath in new_files:
    if not os.path.exists(fpath):
        print(f"[ERROR] File does not exist: {fpath}")
        failed.append((fpath, "File not found"))
        continue
    
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract solution code blocks
    sol_blocks = re.findall(r'::: \{#[a-zA-Z0-9_]+_solution[^\}]*\}\n```(?:python|\{\.python[^\}]*\})\n(.*?)```\n:::', content, re.DOTALL)
    if not sol_blocks:
        print(f"[WARN] No solution block found in: {fpath}")
        failed.append((fpath, "No solution block"))
        continue
        
    for idx, block in enumerate(sol_blocks):
        total_tested += 1
        try:
            # We need standard imports
            test_env = {}
            exec("import numpy as np", test_env, test_env)
            exec(block, test_env, test_env)
            print(f"[PASS] {fpath} - Solution block {idx+1}")
        except Exception as e:
            print(f"[FAIL] {fpath} - Solution block {idx+1}: {e}")
            failed.append((fpath, str(e)))

print(f"\n==========================================")
print(f"VERIFICATION RESULT: {total_tested - len(failed)} / {total_tested} PASSED")
print(f"==========================================")
if failed:
    for f, err in failed:
        print(f"FAILED: {f} -> {err}")
    sys.exit(1)
else:
    print("ALL 18 LESSON SOLUTION CODES EXECUTED & PASSED VERIFICATION!")
    sys.exit(0)
