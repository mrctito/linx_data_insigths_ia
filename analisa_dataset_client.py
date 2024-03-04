import ast
import base64
import io
import os
import pandas as pd
import streamlit as st
import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from st_aggrid import AgGrid

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document, LLMResult
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.prompts import PromptTemplate
from analisa_dataset_service_pandas import svc_analisar_dataset_pandas

from analise_dataset_service_chain import svc_analisar_dataset_chain


def setup_style():
    st.markdown("""
    <style>
        .line-separator {
            border-left: 1px solid #000;
            height: 100%;
            margin: 0 16px;
            color: grey;
        }
    </style>
    """, unsafe_allow_html=True)


def executar_analise_dataset():

    st.title("Análise de Dataset")
    setup_style()

    if "data_frame" not in st.session_state:
        st.session_state.data_frame = None

    if "analise" not in st.session_state:
        st.session_state.analise = None

    if st.session_state.data_frame is None:
        vendas_json = 'vendas.json'
        df = pd.read_json(vendas_json, orient='records')
        st.session_state.data_frame = df

    if st.session_state.data_frame is not None:
        AgGrid(st.session_state.data_frame)

        pergunta = st.selectbox('Escolha uma pergunta, ou digite a sua abaixo:', [
            "Quais são os 3 nomes de produtos mais vendidos?",
            "Qual o mês com mais vendas?",
            "Qual meio de pagamento mais usado?",
            "Existe uma correlação entre o tipo de entrega e a rapidez da conclusão do pedido?",
            "Quais produtos são mais vendidos e qual é o impacto nas vendas de diferentes categorias de produtos?",
            "Como as vendas e os descontos variam ao longo do tempo?"
        ])  

        query = st.text_area(label="Digite sua pergunta aqui:", value=pergunta)

        opcao = st.selectbox(
            'Escolha uma técnica a ser utilizada:',
            ('PANDAS', 'CHAIN')
        )

        if st.button("Executar análise", key="executar_analise"):
            if query and (query is not None) and (len(query) > 0) and (st.session_state.data_frame is not None):
                df = st.session_state.data_frame
                if opcao == 'CHAIN':
                    tabela_json_str = df.to_json(orient='table')
                    st.session_state.analise = svc_analisar_dataset_chain(tabela_json_str, query)
                if opcao == 'PANDAS':
                    json_str = df.to_json(orient='records')
                    st.session_state.analise = svc_analisar_dataset_pandas(json_str, query)

            if st.session_state.analise:
                st.write(st.session_state.analise)