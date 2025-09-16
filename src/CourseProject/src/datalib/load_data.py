from .MoviC import MOVIC
from torch.utils.data import DataLoader
from CONFIG import CONFIG 

# import yaml

# Read the YAML config file
# def read_config(config_path):
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#     return config

def load_data(path, split="train"):
    """
    Loading a dataset given the parameters

    Args:
    -----
    path: string
        path to the dataset to load
    split: string
        Split from the dataset to obtain (e.g., 'train' or 'test')

    Returns:
    --------
    dataset: torch dataset
        Dataset loaded given specifications from exp_params
    """
    dataset = MOVIC(path, split=split)
    return dataset


def build_data_loader(split='train'):
    """
    Fitting a data loader for the given dataset

    Args:
    -----
    dataset: torch dataset
        Dataset (or dataset split) to fit to the DataLoader
    batch_size: integer
        number of elements per mini-batch
    shuffle: boolean
        If True, mini-batches are sampled randomly from the database
    """

    # config = read_config(config_path)
    batch_size = CONFIG['batch_size']
    shuffle = CONFIG[split]['shuffle']
    num_workers = CONFIG["num_workers"]
    path = CONFIG['dataset_path']
    
    dataset = load_data(path,split)
    
    data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    return data_loader