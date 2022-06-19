cd ../../

python main.py \
--do_train \
--exp_name HAN_w_freeze_w2e \
--modelname HAN \
--epochs 100 \
--batch_size 256 \
--lr 3e-3 \
--use_pretrained_word_embed \
--freeze_word_embed \
--max_vocab_size 50000 \
--max_sent_len 16 \
--max_word_len 64 \
--use_saved_data \
--log_interval 10 \
--use_wandb