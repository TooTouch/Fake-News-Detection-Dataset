cd ../../

python main.py \
--do_train \
--exp_name BTS \
--modelname bts \
--pretrained_name 'klue/bert-base' \
--tokenizer 'bert' \
--num_training_steps 20 \
--use_saved_data \
--batch_size 8 \
--use_scheduler \
--lr 1e-5 \
--max_word_len 512 \
--log_interval 1 \
--eval_interval 5 \
--use_wandb