cd ../../

python main.py \
--do_train \
--exp_name BTS \
--modelname BTS \
--pretrained_name 'klue/bert-base' \
--tokenizer 'bert' \
--epochs 10 \
--use_saved_data \
--batch_size 8 \
--use_scheduler \
--lr 3e-3 \
--max_word_len 512 \
--log_interval 10 \
--use_wandb