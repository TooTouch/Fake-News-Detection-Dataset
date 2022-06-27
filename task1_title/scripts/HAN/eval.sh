cd ../../

exp_name=$1

python main.py \
--do_test \
--exp_name $exp_name \
--modelname $exp_name \
--pretarined \
--batch_size 256 \
--max_vocab_size 50000 \
--max_sent_len 16 \
--max_word_len 64 \
--use_saved_data \
--log_interval 10

