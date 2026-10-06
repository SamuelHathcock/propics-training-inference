curl http://inference-first-try-babyyy-predictor-default.tenant-4b0759-dev.knative.chi.coreweave.com \
    -d '{"prompt": "photo of demoura", "parameters": {"seed": 42, "width": 512, "height": 512}}' \
    --output testies.png