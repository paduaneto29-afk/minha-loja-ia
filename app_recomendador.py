# app_recomendador.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from PIL import Image

class RecomendadorEstilo:
    def __init__(self):
        self.cores = {
            'vermelho': [1,0,0], 'azul': [0,0,1], 'preto': [0,0,0],
            'branco': [1,1,1], 'rosa': [1,0.75,0.8], 'bege': [0.96,0.8,0.69]
        }
        self.estilos = ['casual', 'elegante', 'esportiva', 'boho', 'glamour']
    
    def vetorizar_produto(self, cor, estilo, tamanho, preco, estacao):
        cor_vec = self.cores.get(cor.lower(), [0.5,0.5,0.5])
        estilo_vec = [1 if estilo == s else 0 for s in self.estilos]
        return np.array(cor_vec + estilo_vec + [tamanho/50, preco/500, 1 if 'verao' in estacao.lower() else 0])
    
    def recomendar(self, produto_base, estoque_df):
        vec_base = self.vetorizar_produto(
            produto_base['cor'], produto_base['estilo'], 
            produto_base['tamanho'], produto_base['preco'], produto_base['estacao']
        )
        
        estoque_df['similaridade'] = cosine_similarity(
            vec_base.reshape(1,-1), 
            np.array([self.vetorizar_produto(r['cor'], r['estilo'], r['tamanho'], r['preco'], r['estacao']) 
                     for _, r in estoque_df.iterrows()])
        )[0]
        
        return estoque_df.nlargest(5, 'similaridade')

# Interface Streamlit
def main():
    st.title("🛍️ Recomendador de Estilo IA")
    st.markdown("**Recomendações personalizadas para suas clientes!**")
    
    # Upload de estoque
    uploaded_file = st.file_uploader("📁 Upload CSV do Estoque", type='csv')
    
    if uploaded_file:
        estoque = pd.read_csv(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Produto Selecionado")
            produto = {
                'cor': st.selectbox("Cor", ['vermelho', 'azul', 'preto', 'rosa', 'bege']),
                'estilo': st.selectbox("Estilo", ['casual', 'elegante', 'esportiva', 'boho', 'glamour']),
                'tamanho': st.slider("Tamanho", 36, 46, 40),
                'preco': st.slider("Preço R$", 50, 500, 150),
                'estacao': st.selectbox("Estação", ['verao', 'inverno', 'outono'])
            }
        
        if st.button("🔮 Gerar Recomendações"):
            rec = RecomendadorEstilo()
            recomendacoes = rec.recomendar(produto, estoque)
            
            st.subheader("💎 Recomendações Perfeitas")
            for _, item in recomendacoes.iterrows():
                col1, col2, col3 = st.columns([1,3,1])
                with col1:
                    st.write(f"**{item['nome']}**")
                with col2:
                    st.caption(f"{item['cor'].title()} | {item['estilo'].title()} | R${item['preco']}")
                with col3:
                    st.metric("Similaridade", f"{item['similaridade']:.1%}")

if __name__ == "__main__":
    main()
