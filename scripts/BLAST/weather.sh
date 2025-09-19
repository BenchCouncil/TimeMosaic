model_name=Mosaic
# Test MSE: 0.3988660 Test MAE: 0.4238716

python -u run_blast.py \
  --task_name Exp_BLAST \
  --is_training 0 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id BLAST_512_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --batch_size 256 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --num_latent_token 8 \
  --patch_len_list '[8,16,32,64]' \
  --n_heads 8 \
  --des 'Exp' \
  --itr 1

python -u run_blast.py \
  --task_name Exp_BLAST \
  --is_training 0 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id BLAST_512_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --batch_size 256 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --num_latent_token 8 \
  --patch_len_list '[8,16,32,64]' \
  --n_heads 8 \
  --des 'Exp' \
  --itr 1

python -u run_blast.py \
  --task_name Exp_BLAST \
  --is_training 0 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id BLAST_512_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 336 \
  --batch_size 256 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --num_latent_token 8 \
  --patch_len_list '[8,16,32,64]' \
  --n_heads 8 \
  --des 'Exp' \
  --itr 1

python -u run_blast.py \
  --task_name Exp_BLAST \
  --is_training 0 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id BLAST_512_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 720 \
  --batch_size 256 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --num_latent_token 8 \
  --patch_len_list '[8,16,32,64]' \
  --n_heads 8 \
  --des 'Exp' \
  --itr 1