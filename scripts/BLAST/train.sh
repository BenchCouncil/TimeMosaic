model_name=Mosaic

python -u run_blast.py \
  --task_name Exp_BLAST \
  --is_training 1 \
  --root_path ./dataset/BLAST/ \
  --model_id BLAST_512_720 \
  --model $model_name \
  --data BLAST \
  --use_multi_gpu \
  --devices 0,1 \
  --test_root_path ./dataset/ETT-small/ \
  --test_data_path ETTh1.csv\
  --test_data ETTh1 \
  --features M \
  --train_epochs 1 \
  --batch_size 256 \
  --seq_len 512 \
  --pred_len 720 \
  --test_pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --segment 16 \
  --num_latent_token 8 \
  --patch_len_list '[8,16,32,64]' \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1 > BLAST_512_720_e1_ep10_lat8.log