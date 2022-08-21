import re
import numpy as np
import pandas as pd
import json
import os
import argparse

from sklearn.model_selection import train_test_split



def _extract_article_info(doc):
    article_info = {}
    
    for k in ['category','media_type','media_name','size','title','text']:
        if doc['id'] not in article_info.keys():
            article_info[doc['id']] = {}

        if k == 'text':
            article_info[doc['id']][k] = [t['sentence'] for t in sum(doc[k], [])]
        else:
            article_info[doc['id']][k] = doc[k]

    return article_info


def extract_article_info(data):
    articles = {}
    for doc in data['documents']:
        article = _extract_article_info(doc)
        articles.update(article)
    return articles



def remove_reportor_email_in_text(articles):
    """
    기사 본문에 기사의 부제목과 기자의 '이름 이메일주소'가 함께 들어간 경우 제외
    
    ex)
    ['ha당 조사료 400만원…작물별 차등 지원',
     '이성훈 sinawi@hanmail.net',
     ...
    ]
    
    """
    reg = re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')

    for article in articles.values():
        try:
            if reg.match(article['text'][1].split(' ')[1]) != None:
                article['text'] = article['text'][2:]
        except:
            continue

    return articles



def _change_article_title(article1, article2):
    title1 = article1['title'].copy()
    title2 = article2['title'].copy()
    
    article1['title'] = title2
    article2['title'] = title1


def change_article_title(articles, fake_id):
    paired_fake_id = list(zip(fake_id[::2], fake_id[1::2]))
    for fake1, fake2 in paired_fake_id:
        _change_article_title(articles[fake1], articles[fake2])

    return articles


def _change_article_text(article1, article2, min=2, max=5):
    select_len = np.random.randint(low=min, high=max, size=1)[0]

    text1 = article1['text'].copy()
    text2 = article2['text'].copy()
    
    article1['text'][-select_len:] = text2[-select_len:]
    article2['text'][-select_len:] = text1[-select_len:]

    return article1, article2

def change_article_text(articles, fake_id, min=2, max=5):
    paired_fake_id = list(zip(fake_id[::2], fake_id[1::2]))
    for fake1, fake2 in paired_fake_id:
        articles[fake1], articles[fake2] = _change_article_text(articles[fake1], articles[fake2], min=min, max=max)

    return articles


def define_real_fake_articles(articles):
    np.random.seed(42)
    fake_size = len(articles)//2

    fake_id = np.random.choice(
        list(articles.keys()), 
        size    = fake_size - 1 if fake_size % 2 else fake_size, 
        replace = False
    )
    real_id = list(set(articles.keys()) - set(fake_id))

    real_fake_df = pd.concat([
        pd.DataFrame({'id':fake_id,'label':['fake']*len(fake_id)}),
        pd.DataFrame({'id':real_id,'label':['real']*len(real_id)})
    ], axis=0)

    return real_fake_df



def preprocessing(articles):
    
    articles = remove_reportor_email_in_text(articles)

    return articles




def split_train_valid(articles, real_fake_df, train_ratio):
    
    train_df, valid_df = train_test_split(real_fake_df, train_size=train_ratio, stratify=real_fake_df['label'])
    
    train_articles = {}
    for id in train_df['id']:
        train_articles[id] = articles[id]

    valid_articles = {}
    for id in valid_df['id']:
        valid_articles[id] = articles[id]

    return train_articles, valid_articles, train_df, valid_df




def make_dataset(args):
    data = json.load(open(os.path.join(args.datadir,f'{args.data}_original.json'),'r'))

    articles = extract_article_info(data)

    # preprocessing
    articles = preprocessing(articles)
    
    # define real and fake articles
    real_fake_df = define_real_fake_articles(articles)
    
    # make fake articles
    if args.target == 'title':
        articles = change_article_title(articles, real_fake_df[real_fake_df['label']=='fake']['id'])
    elif args.target == 'text':
        articles = change_article_text(articles, real_fake_df[real_fake_df['label']=='fake']['id'])

    if args.data == 'train':
        # split train and validation
        train_articles, valid_articles, train_df, valid_df = split_train_valid(articles, real_fake_df, args.train_ratio)
        return (train_articles, valid_articles), (train_df, valid_df)

    elif args.data == 'valid':
        return articles, real_fake_df
    


    
def get_args(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default='train',choices=['train','valid'])
    parser.add_argument('--datadir',type=str,default='./data/origin/',help='data directory')
    parser.add_argument('--savedir',type=str,default='./data/task1/',help='save directory')

    parser.add_argument('--train_ratio',type=float,default=0.8,help='train ratio')
    parser.add_argument('--target',type=str,default='title',choices=['title','text'],help='select target')

    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    # make save folder
    os.makedirs(args.savedir, exist_ok=True)

    articles, real_fake_df = make_dataset(args)

    if args.data == 'train':
        real_fake_df[0].to_csv(os.path.join(args.savedir, 'train_info.csv'), index=False)
        real_fake_df[1].to_csv(os.path.join(args.savedir, 'valid_info.csv'), index=False)
        print(f'Save train and valid info to {args.savedir}')

        json.dump(articles[0], open(os.path.join(args.savedir, 'train.json'),'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        json.dump(articles[1], open(os.path.join(args.savedir, 'valid.json'),'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        print(f'Save train and valid articles to {args.savedir}')

    elif args.data == 'valid':
        real_fake_df.to_csv(os.path.join(args.savedir, 'test_info.csv'), index=False)
        print(f'Save test info to {args.savedir}')

        json.dump(articles, open(os.path.join(args.savedir, 'test.json'),'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        print(f'Save test articles to {args.savedir}')

    