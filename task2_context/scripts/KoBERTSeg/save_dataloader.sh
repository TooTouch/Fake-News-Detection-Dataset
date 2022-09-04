cd ../../

python save_dataloader.py \
    --modelname KoBERTSeg \
    --batch_size 128 \
    --window_size 3 \
    --max_word_len 512