cd ../../

python main.py \
--do_train \
--exp_name KoBERTSeg \
--modelname kobertseg \
--num_training_steps 10000 \
--batch_size 8 \
--use_scheduler \
--lr 1e-5 \
--max_word_len 512 \
--log_interval 1 \
--eval_interval 2500 \
--use_wandb