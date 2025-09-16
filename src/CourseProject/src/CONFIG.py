"""
Global configurations
"""

import os

CONFIG = {'dataset_path': '/home/nfs/inf6/data/datasets/MOVi/movi_c/',

        'batch_size': 64,

         'num_workers': 0,
 
         'train':{
             'shuffle': True,
             'transforms': None
             },
         'validation':{
             'shuffle': False,
             'transforms': None
             },
         'num_epochs':100
                
}