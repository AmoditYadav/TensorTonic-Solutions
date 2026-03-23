import numpy as np
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k=recommended[:k]
    #hits=item in top_k that are relevant
    hits=sum(1 for item in top_k if item in relevant)
    precision=hits/k if k>0 else 0.0
    recall=hits/len(relevant) if len(relevant)>0 else 0.0
   

    return [precision,recall]