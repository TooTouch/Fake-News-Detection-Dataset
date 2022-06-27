cd ../../

exp_name=$1

python main.py \
--do_test \
--exp_name $exp_name \
--modelname $exp_name \
--pretrained \
--batch_size 256 \
--use_saved_data \
--max_vocab_size 50000 \
--max_word_len 1000 \
--dims 128 \
--log_interval 10