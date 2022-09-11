from torch.utils.data import DataLoader

def create_dataset(name, data_path, split, tokenizer, vocab, **kwargs):
    dataset = __import__('dataset').__dict__[f'{name}Dataset'](
        tokenizer       = tokenizer,
        vocab           = vocab,
        **kwargs
    )

    dataset.load_dataset(datadir=data_path, split=split)
    dataset.preprocessor()

    return dataset


def create_dataloader(dataset, batch_size, num_workers, shuffle=False):
    dataloader = DataLoader(
        dataset, 
        batch_size  = batch_size, 
        num_workers = num_workers, 
        shuffle     = shuffle
    )

    return dataloader