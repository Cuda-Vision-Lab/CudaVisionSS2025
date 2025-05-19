import random
from torchvision import transforms

class RandomTemporalSampling: #---> Good to use but does not preserve the tensor shape!
    """
    Sample every other frame from the sequence, effectively halving the number of frames.
    
    Example:
        Given a sequence of 40 frames, it will return 20 frames.
    """
    def __init__(self, slicing_step):
        # We slice the frames with a step size of slicing_step to include more semantically important sequence frames
        # This is a good trade-off between the number of frames and the semantic importance of the frames
        
        self.slicing_step = slicing_step

    def __call__(self, frames):
        # Always use step size 2 if there are enough frames
        if frames.size(0) > 2:
            return frames[::self.slicing_step]
        return frames

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
def get_train_transforms(slicing_step):
    """
    Returns the composition of transforms for training.
    """
    return transforms.Compose([
        # Spatial augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        # Temporal augmentations
        RandomTemporalSampling(slicing_step),
        RandomTemporalReverse(p=0.3),
    ])

def get_test_transforms(slicing_step):
    """
    Returns the composition of transforms for testing.
    """
    return transforms.Compose([
        RandomTemporalSampling(slicing_step)
    ])