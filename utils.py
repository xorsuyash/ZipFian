import torch 
from torch.autograd import Variable 
import numpy as np 

def get_probablity(class_id,set_size):

    class_prob=(np.log(class_id+2)-np.log(class_id+1))/np.log(set_size+1)

    return class_prob 
def make_sampling_array(range_max):
    """ Creates and populates the array from which the fake labels are sampled during the NCE loss calculation."""
    # Get class probabilities
    print('Computing the Zipfian distribution probabilities for the corpus items.')
    class_probs = {class_id: get_probablity(class_id, range_max) for class_id in range(range_max)}

    print('Generating and populating the sampling array. This may take a while.')
    # Generate empty array
    sampling_array = np.zeros(int(1e8))
    # Determine how frequently each index has to appear in array to match its probability
    class_counts = {class_id: int(np.round((class_probs[class_id] * 1e8))) for class_id in range(range_max)}
    assert(sum(list(class_counts.values())) == 1e8), 'Counts don\'t add up to the array size!'

    # Populate sampling array
    pos = 0
    for key, value in class_counts.items():
        while value != 0:
            sampling_array[pos] = key
            pos += 1
            value -= 1
    
    return sampling_array, class_probs



