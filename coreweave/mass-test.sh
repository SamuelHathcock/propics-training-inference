argo submit coreweave/inference-workflow.yaml \
    -p run_name="abby" \
    -p dataset="dataset" \
    -p templates_path="test-template.json" &

argo submit coreweave/inference-workflow.yaml \
    -p run_name="alejandro" \
    -p dataset="dataset" \
    -p templates_path="test-template.json" &

argo submit coreweave/inference-workflow.yaml \
    -p run_name="black-guy" \
    -p dataset="dataset" \
    -p templates_path="test-template.json" &

argo submit coreweave/inference-workflow.yaml \
    -p run_name="kait" \
    -p dataset="dataset" \
    -p templates_path="test-template.json" &

argo submit coreweave/inference-workflow.yaml \
    -p run_name="logan" \
    -p dataset="dataset" \
    -p templates_path="test-template.json" &

