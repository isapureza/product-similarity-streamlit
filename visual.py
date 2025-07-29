import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from umap import UMAP

with open('new_product_embeddings2.pkl', 'rb') as f:
    product_embeddings = pickle.load(f)

#Combine helper function 
def combine_embeddings(title_emb, ing_emb, desc_emb, w1, w2, w3, ing_present=True):
  if not ing_present: #ing_present == False
    #Normalize weights without ingredients
    total = w1 + w3
    w1 = w1/ total
    w3 = w3 / total
    w2 = 0.0 #no ingredient weight

  #Weighted combination of title, ingredients, and description embeddings.
  combined = w1 * title_emb + w2 * ing_emb + w3 * desc_emb
  return combined / np.linalg.norm(combined)

def get_top_similar_products(selected_title, product_embeddings, w1, w2, w3, top_n=5, alpha=0.3, beta=0.3):
    '''
    alpha: weight for retention
    beta: weight for sold_count
    '''
    
    #Find the selected product
    selected = next((prod for prod in product_embeddings if prod['title'] == selected_title), None)
    if not selected:
        raise ValueError(f"Product with title '{selected_title}' not found.")
    
    #Combine embeddings for selected product
    query_emb = combine_embeddings(
        selected['title_emb'], selected['ing_emb'], selected['desc_emb'], w1, w2, w3
    ).reshape(1, -1) #shape (1, embedding_dim)

    '''
    Calculate a popularity score combining retention and sold count (scaled between 0 and 1)
    Adjust the cosine similarity by boosting it with a weighted popularity factor
    Tune the weights for popularity influence (to control how much it biases results)
    '''

    #Normalize retention and sold_count across dataset for scaling
    retentions = np.array([p['90 Day Repurchase Rate'] for p in product_embeddings])
    sold_counts = np.array([p['sold_count'] for p in product_embeddings])

    ret_min, ret_max = retentions.min(), retentions.max()
    sold_min, sold_max = sold_counts.min(), sold_counts.max()

    def normalize(value, vmin, vmax):
       return (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0

    similarities = []

    for prod in product_embeddings:
        if prod['title'] == selected_title:
            continue #skip self
        emb = combine_embeddings(prod['title_emb'], prod['ing_emb'], prod['desc_emb'], w1, w2, w3)

        #Compute cosine sim
        sim = cosine_similarity(query_emb, emb.reshape(1, -1))[0][0]

        #Normalize popularity features
        norm_ret = normalize(prod['90 Day Repurchase Rate'], ret_min, ret_max)
        norm_sold = normalize(prod['sold_count'], sold_min, sold_max)

        #Compute popularity boost (between 1.0 and 1 + weighted sum)
        popularity_boost = 1 + alpha * norm_ret + beta * norm_sold

        #Adjust sim scoe by popularity
        adjusted_score = sim * popularity_boost

        similarities.append({
            'title': prod['title'],
            'vendor': prod['vendor'],
            'similarity': sim,
            'retention': prod['90 Day Repurchase Rate'],
            'sold_count': prod['sold_count'],
            'adjusted_score': adjusted_score
        })

    #Sort by similarity descending
    sorted_similar = sorted(similarities, key=lambda x: x['adjusted_score'], reverse=True)
    
    #Return top N
    return sorted_similar[:top_n]


w1, w2, w3 = 0.33, 0.33, 0.33
all_embs = np.array([
   combine_embeddings(p['title_emb'], p['ing_emb'], p['desc_emb'], w1, w2, w3)
    for p in product_embeddings
])

# Run UMAP
umap_model = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
umap_2d = umap_model.fit_transform(all_embs)

# Save to file
df_umap = pd.DataFrame({
    'title': [p['title'] for p in product_embeddings],
    'brand': [p['vendor'] for p in product_embeddings],
    'x': umap_2d[:, 0],
    'y': umap_2d[:, 1],
})
df_umap.to_pickle('umap_products.pkl')