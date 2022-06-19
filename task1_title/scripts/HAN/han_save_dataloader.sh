cd ../../

python save_dataloader.py \
    --modelname HAN \
    --batch_size 256 \
    --max_sent_len 16 \
    --max_word_len 64 \
    --max_vocab_size 50000