cd ../../

exp_name=$1

python main.py \
--do_test \
--pretrained_path ./saved_model/$exp_name/best_model.pt \
--modelname FNDNet \
--batch_size 256 \
--use_pretrained_word_embed \
--use_saved_data \
--max_vocab_size 50000 \
--max_word_len 1000 \
--dims 128 \
--log_interval 10