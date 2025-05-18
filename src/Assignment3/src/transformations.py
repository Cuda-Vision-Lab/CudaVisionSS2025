import random
from torchvision import transforms

# class RandomTemporalSampling: ---> Good to use but does not preserve the tensor shape!
#     """
#     Randomly sample frames from the sequence with different step sizes.
    
#     Args:
#         p (float): probability of applying the sampling. Default: 0.5
#     """
#     def __init__(self, p=0.5):
#         self.p = p
        
#     def __call__(self, frames):
#         if random.random() < self.p:
#             step = random.choice([1, 2])
#             if step > 1 and frames.size(0) > step:
#                 return frames[::step]
#         return frames

class RandomTemporalReverse:
    """
    Randomly reverse the order of frames in the sequence.
    
    Args:
        p (float): probability of applying the reversal. Default: 0.5
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, frames):
        if random.random() < self.p:
            return frames.flip(0)  # Reverse temporal dimension
        return frames


# Combined transforms
def get_train_transforms():
    """
    Returns the composition of transforms for training.
    """
    return transforms.Compose([
        # Spatial augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        # Temporal augmentations
        RandomTemporalReverse(p=0.3),
    ])

