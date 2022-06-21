cd ../../

exp_name=$1

python main.py \
--do_test \
--modelname HAN \
--pretrained_path ./saved_model/$exp_name/best_model.pt \
--batch_size 256 \
--use_pretrained_word_embed \
--max_vocab_size 50000 \
--max_sent_len 16 \
--max_word_len 64 \
--use_saved_data \
--log_interval 10

