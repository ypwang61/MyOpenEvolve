# # Phase 1: Initial exploration
# python openevolve-run.py examples/circle_packing/initial_program.py \
#   examples/circle_packing/evaluator.py \
#   --config examples/circle_packing/config_phase_1.yaml \
#   --iterations 100 #\
#   # --checkpoint examples/circle_packing/openevolve_output/checkpoints/checkpoint_100/openevolve_output/checkpoints/checkpoint_100

# echo "Phase 1 completed, starting Phase 2"

# # Phase 2: Breaking through the plateau
# python openevolve-run.py examples/circle_packing/openevolve_output/checkpoints/checkpoint_100/best_program.py \
#   examples/circle_packing/evaluator.py \
#   --config examples/circle_packing/config_phase_2.yaml \
#   --iterations 200


# # Phase 2: Breaking through the plateau
# python openevolve-run.py examples/circle_packing/openevolve_output/checkpoints/checkpoint_100/openevolve_output/checkpoints/checkpoint_100/best_program.py \
#   examples/circle_packing/evaluator.py \
#   --config examples/circle_packing/config_phase_3.yaml \
#   --iterations 300

# Phase 2: Breaking through the plateau
python openevolve-run.py examples/circle_packing/openevolve_output/checkpoints/checkpoint_100/openevolve_output/checkpoints/checkpoint_100/openevolve_output/checkpoints/checkpoint_250/best_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_4.yaml \
  --iterations 200