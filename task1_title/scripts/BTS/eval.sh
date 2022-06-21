cd ../../

exp_name=$1

python main.py \
--do_test \
--modelname BTS \
--pretrained_name 'klue/bert-base' \
--pretrained_path ./saved_model/$exp_name/best_model.pt \
--tokenizer 'bert' \
--use_saved_data \
--batch_size 8 \
--max_word_len 512 \
--log_interval 10