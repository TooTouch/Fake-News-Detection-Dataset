
from tqdm.auto import tqdm
import torch 
import os 
import yaml
import argparse 

from dataset import create_tokenizer, create_dataset, create_dataloader

def save(split, dataloader, savedir):
    
    doc_dict = {}
    label_list = []
    for i, (doc, label) in enumerate(tqdm(dataloader, desc=split)):
        if len(doc_dict) == 0:
            for k in doc.keys():
                doc_dict[k] = []
            
        for k in doc.keys():
            doc_dict[k].append(doc[k])
        label_list.append(label)

    for k in doc_dict.keys():
        doc_dict[k] = torch.cat(doc_dict[k])
    label_list = torch.cat(label_list)

    torch.save({'doc':doc_dict, 'label':label_list}, os.path.join(savedir,f'{split}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')
    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    # save directory
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['RESULT']['dataname'])
    os.makedirs(savedir, exist_ok=True)

    # tokenizer
    tokenizer, word_embed = create_tokenizer(
        name            = cfg['TOKENIZER']['name'], 
        vocab_path      = cfg['TOKENIZER'].get('vocab_path', None), 
        max_vocab_size  = cfg['TOKENIZER'].get('max_vocab_size', None)
    )

    for split in ['train','validation','test']:
        dataset = create_dataset(
            name           = cfg['DATASET']['name'], 
            data_path      = cfg['DATASET']['data_path'], 
            split          = split, 
            tokenizer      = tokenizer, 
            saved_data_path = cfg['DATASET']['saved_data_path'],
            **cfg['DATASET']['PARAMETERS']
        )
        
        dataloader = create_dataloader(
            dataset     = dataset, 
            batch_size  = cfg['TRAIN']['batch_size'], 
            num_workers = cfg['TRAIN']['num_workers'],
            shuffle     = False
        )
   
        # save
        save(split, dataloader, savedir)
        