import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import plotly.express as px
from visual import get_top_similar_products

#Load data
with open('new_product_embeddings.pkl', 'rb') as f:
    product_embeddings = pickle.load(f)

with open ('umap_products.pkl', 'rb') as f:
    umap_embeddings = pickle.load(f)

#Convert UMAP embeddings to DF
umap_df = pd.DataFrame(umap_embeddings, columns=["x", "y"])
titles = [prod['title'] for prod in product_embeddings]
umap_df['title'] = titles

#Title and search
st.title("Product Explorer")

selected_title = st.text_input("Search for a product title:")
if selected_title:
    matches = [title for title in titles if selected_title.lower() in title.lower()]
    if matches: 
        selected = st.selectbox("Select a matching product:", matches)
        
        st.sidebar.header("Adjust Weights")
        w1 = st.sidebar.slider("Weight for Title Embedding", 0.0, 1.0, 0.4)
        w2 = st.sidebar.slider("Weight for Ingredients Embedding", 0.0, 1.0, 0.3)
        w3 = st.sidebar.slider("Weight for Description Embedding", 0.0, 1.0, 0.3)
        alpha = st.sidebar.slider("Weight for Retention", 0.0, 1.0, 0.3)
        beta = st.sidebar.slider("Weight for Sold Count", 0.0, 1.0, 0.3)
        
        total = w1 + w2 + w3
        if total > 0:
            w1 /= total
            w2 /= total
            w3 /= total
        else:
            w1, w2, w3 = 0.33, 0.33, 0.34
      
        top_similar = get_top_similar_products(
            selected_title=selected,
            product_embeddings=product_embeddings, w1=0.4, w2=0.3, w3=0.3,
            alpha=0.5, beta=0.5,
            top_n=5
        )

        st.subheader("Most Similar Products:")
        for sim in top_similar:
            st.write(f"• {sim['title']} (Score: {sim['adjusted_score']:.4f})")

        st.subheader("UMAP Visualization")
        fig = px.scatter(
            umap_df,
            x="x",
            y="y",
            hover_data=["title"], 
            title="UMAP of Project Embeddings"
        )
        selected_index=titles.index(selected)
        fig.add_scatter(x=[umap_df.iloc[selected_index]["x"]],
                        y=[umap_df.iloc[selected_index]["y"]],
                        mode='markers',
                        marker=dict(size=12, color='red'),
                        name="Selected Product")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No matching titles found.")

