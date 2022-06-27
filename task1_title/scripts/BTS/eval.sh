cd ../../

exp_name=$1

python main.py \
--do_test \
--exp_name $exp_name \
--modelname $exp_name \
--pretrained \
--pretrained_name 'klue/bert-base' \
--tokenizer 'bert' \
--use_saved_data \
--batch_size 8 \
--max_word_len 512 \
--log_interval 10