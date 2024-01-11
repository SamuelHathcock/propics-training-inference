accelerate launch finetuner.py \
  --dev \
  --user_id="iyLkPXiD27ajYzryV3Cd8Pti3En2" \
  --run_name="andrew-priorpres_weight_0.05_with-checkpointing-and_diff_upper_lower_bound" \
  --pretrained_model_name_or_path="/run/determined/workdir/sd-finetune-data/models/samiam/sd-v1-5_vae-pruned"  \
  --instance_data_dir="/run/determined/workdir/sd-finetune-data/datasets/andrew/training_base" \
  --validation_data_dir="/run/determined/workdir/sd-finetune-data/andrew/validation" \
  --class_data_dir="/run/determined/workdir/sd-finetune-data/class-imgs/male-all" \
  --val_step_interval=10 \
  --output_dir="/run/determined/workdir/sd-finetune-data/finetunes/andrew-priorpres_weight_0.05_with-checkpointing-and_diff_upper_lower_bound" \
  --instance_prompt="headshot photo of demoura" \
  --class_prompt="headshot photo of person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --final_learning_rate=3e-7 \
  --lr_decay_rate=40 \
  --lr_scheduler="constant" \
  --save_checkpoints \
  --diff_save_range_lower_bound=0.03 \
  --diff_save_range_upper_bound=0.04 \
  --patience=400 \
  --burn_in=100 \
  --lr_warmup_steps=0 \
  --max_train_steps=150 \
  --mixed_precision=fp16 \
  --checkpointing_steps=100000 \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder_steps=6000 \
  --train_text_encoder \
  --initial_LR_after_text_encoder=1e-6 \
  --report_to="wandb" \
  --with_prior_preservation --prior_loss_weight=0.05
  
  # --instance_data_dir="/run/determined/workdir/sd-finetune-data/datasets/andrew/training_base" \
  # --validation_data_dir="/run/determined/workdir/sd-finetune-data/andrew/validation" \
