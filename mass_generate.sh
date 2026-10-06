# python mass_generate.py \
#     --generation_id="gen-20230611" \
#     --user_id="iyLkPXiD27ajYzryV3Cd8Pti3En2" \
#     --pretrained_model_name_or_path="../users/iyLkPXiD27ajYzryV3Cd8Pti3En2/finetunes/initial" \
#     --embeddings_path="embeddings" \
#     --gen_count=5 \
#     --templates_path="premium-template-female.json" \
#     --output_dir="results/iyLkPXiD27ajYzryV3Cd8Pti3En2/inference" \
#     --dev

python mass_generate_dev.py \
    --pretrained_model_name_or_path="/run/determined/workdir/sd-finetune-data/finetunes/andrew-priorpres_weight_0.05_with-checkpointing-and_diff_upper_lower_bound-1541_steps" \
    --embeddings_path="embeddings" \
    --gen_count=6 \
    --templates_path="templates.json" \
    --output_dir="/run/determined/workdir/sd-finetune-data/results/andrew-priorpres_weight_0.05_with-checkpointing-and_diff_upper_lower_bound" \
    --facial_filter_basis_img="kait.jpg"
