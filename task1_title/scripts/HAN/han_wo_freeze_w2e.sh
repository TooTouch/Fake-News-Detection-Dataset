cd ../../

python main.py \
--do_train \
--exp_name HAN_limit \
--modelname HAN \
--epochs 30 \
--batch_size 256 \
--use_pretrained_word_embed \
--max_vocab_size 50000 \
--max_sent_len 16 \
--max_word_len 64 \
--log_interval 10 \
--use_saved_data \
--use_wandb