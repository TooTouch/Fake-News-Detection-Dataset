import json
from glob import glob
import os
import pandas as pd
import argparse
import yaml

from sklearn.model_selection import train_test_split



def run(datadir: str, savedir: str, split_ratio: list, method: str, seed: int) -> None:
    folders = os.listdir(os.path.join(datadir))
    assert 'Clickbait_Auto' in folders, f'Clickbait_Auto folder does not exist in {datadir}'
    assert 'NonClickbait_Auto' in folders, f'NonClickbait_Auto folder does not exist in {datadir}'

    task1 = glob(os.path.join(datadir,'*/*/*'))
    task1 = [f for f in task1 if 'Clickbait_Direct' not in f]

    filenames = []
    labels = []
    for p in task1:
        filename = p.replace('/Clickbait_Auto',f'/Clickbait_Auto_{method}') if '/Clickbait_Auto' in p else p
        filename = '/'.join(filename.split('/')[-3:])
        
        filenames.append(filename)
        
        label = 'real' if 'NonClickbait_Auto' in p else 'fake'
        labels.append(label)

    info = pd.DataFrame({'filename':filenames,'label':labels})

    # split train, valid, and test
    train_size = int(len(info) * split_ratio[0]/sum(split_ratio))
    valid_size = int(len(info) * split_ratio[1]/sum(split_ratio))

    train_info, test_info = train_test_split(info, train_size=train_size, stratify=info['label'], random_state=seed)
    valid_info, test_info = train_test_split(test_info, train_size=valid_size, stratify=test_info['label'], random_state=seed)


    # save
    train_info.to_csv(os.path.join(savedir,'train_info.csv'),index=False)
    valid_info.to_csv(os.path.join(savedir,'valid_info.csv'),index=False)
    test_info.to_csv(os.path.join(savedir,'test_info.csv'),index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, help='config filename')
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    # define save directory of method type
    cfg['SPLIT']['savedir'] = os.path.join(cfg['SPLIT']['savedir'], cfg['BUILD']['METHOD']['name'])
    os.makedirs(cfg['SPLIT']['savedir'], exist_ok=True)

    run(
        datadir     = cfg['SPLIT']['datadir'],
        savedir     = cfg['SPLIT']['savedir'],
        split_ratio = cfg['SPLIT']['ratio'],
        method      = cfg['BUILD']['METHOD']['name'],
        seed        = cfg['SEED']
    )