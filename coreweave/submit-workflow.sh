argo submit coreweave/finetune-workflow.yaml \
    -p run_name="sam" \
    -p model=samiam/sd-v1-5_vae-pruned \
    -p dataset="datasets/nathan" \
    -p hf_token=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 \
    -p wandb_api_key=c721be99e8377261cc3bfdd5b6b0a668a6583916 \
    -p run_inference=false \
    -p inference_only=false \
    --serviceaccount inference