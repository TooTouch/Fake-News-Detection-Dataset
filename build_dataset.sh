# task1; title
python build_dataset.py --data train --target title --savedir ./data/task1
python build_dataset.py --data valid --target title --savedir ./data/task1

# task2: text
python build_dataset.py --data train --target text --savedir ./data/task2
python build_dataset.py --data valid --target text --savedir ./data/task2