argo submit coreweave/inference-workflow.yaml \
    -p run_name="michael" \
    -p dataset="dataset" \
    -p templates_path="test-template.json"

# argo submit coreweave/inference-workflow.yaml \
#     -p run_name="logan" \
#     -p model=samiam/sd-v1-5_vae-pruned \
#     -p dataset="dataset" \
#     -p templates_path="test-template.json"
