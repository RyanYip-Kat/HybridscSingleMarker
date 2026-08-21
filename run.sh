# 1
uv run scripts/hybrid_pipeline2.py --full --output-dir results/skeleton_full_profile_ft --scoring profile --n-jobs 32 --fine-tune  --label-evidence

uv run scripts/hybrid_pipeline2.py --full --output-dir results/skeleton_full_cell_ft --scoring cell --n-jobs 32 --fine-tune  --label-evidence

# 2
uv run scripts/hybrid_pipeline2.py --full --output-dir results/skeleton_full_profile --scoring profile --n-jobs 32 --no-fine-tune --label-evidence

uv run scripts/hybrid_pipeline2.py --full --output-dir results/skeleton_full_cell --scoring cell --n-jobs 32 --no-fine-tune --label-evidence
