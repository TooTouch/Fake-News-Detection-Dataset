cd ../../

python main.py \
--do_train \
--exp_name FNDNet_w_freeze_w2e \
--modelname FNDNet \
--epochs 100 \
--batch_size 256 \
--lr 1e-3 \
--weight_decay 0. \
--use_pretrained_word_embed \
--freeze_word_embed \
--use_saved_data \
--max_vocab_size 50000 \
--max_word_len 1000 \
--dims 128 \
--log_interval 10 \
--use_wandb