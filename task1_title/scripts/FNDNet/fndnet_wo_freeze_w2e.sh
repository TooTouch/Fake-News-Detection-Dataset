cd ../../

python main.py \
--do_train \
--exp_name FNDNet_wo_freeze_w2e \
--modelname FNDNet \
--num_training_steps 100000 \
--batch_size 256 \
--use_scheduler \
--lr 3e-5 \
--use_pretrained_word_embed \
--use_saved_data \
--max_vocab_size 50000 \
--max_word_len 1000 \
--dims 128 \
--log_interval 10 \
--eval_interval 1000 \
--use_wandb