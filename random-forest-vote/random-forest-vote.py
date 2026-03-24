import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions=np.array(predictions)
    if predictions.size==0:
        return []
    n_samples=predictions.shape[1] #rows
    result=[]

    for i in range(n_samples):
        unique,counts=np.unique(predictions[:,i],return_counts=True)
        max_count=np.max(counts)
        winners=unique[counts==max_count]
        result.append(int(np.min(winners)))

    return result