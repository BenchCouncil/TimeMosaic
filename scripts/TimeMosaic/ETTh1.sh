
model_name=TimeMosaic

python -u run.py \
  --task_name TimeMosaic \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_320_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 320 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 8 \
  --train_epochs 10 \
  --itr 1

python -u run.py \
  --task_name TimeMosaic \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_320_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 320 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 1 \
  --itr 1

python -u run.py \
  --task_name TimeMosaic \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_320_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --channel CDA \
  --seq_len 320 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 16 \
  --itr 1

python -u run.py \
  --task_name TimeMosaic \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_320_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --channel CD \
  --seq_len 320 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --itr 1
