import ast
import base64
import io
import os
import httpx
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


def executar_analisys_dataset():

    st.title("Análise de Dataset")
    setup_style()

    if "data_frame" not in st.session_state:
        st.session_state.data_frame = None

    if "analise" not in st.session_state:
        st.session_state.analise = None

    if st.session_state.data_frame is not None:
        vendas_json = 'vendas.json'
        df = pd.read_json(vendas_json)
        st.session_state.data_frame = df
        AgGrid(st.session_state.data_frame)

        query = st.text_area(label="Digite sua pergunta aqui:", value="quais são os 3 nomes de produtos mais vendidos?")
        opcao = st.selectbox(
            'Escolha uma opção:',
            ('PANDAS', 'CHAIN')
        )

        if st.button("Executar análise", key="executar_analise"):
            if query and (query is not None) and (len(query) > 0) and (st.session_state.data_frame is not None):
                if st.session_state.dataset_id == None:

                    json_str = df.to_json(orient='records')
                    if opcao == 'CHAIN':
                        st.session_state.analise = svc_analisar_dataset_chain(json_str, query)
                    if opcao == 'PANDAS':
                        st.session_state.analise = svc_analisar_dataset_pandas(json_str, query)


            if st.session_state.analise:
                st.write(st.session_state.analise)

                try:
                    response_dict = json.loads(st.session_state.analise)
                except Exception as e:
                    print(f"Ocorreu um erro em 'json.loads(st.session_state.analise)': {e}")
                    return

                # Check if the response is an answer.
                if "answer" in response_dict:
                    st.write(response_dict["answer"])

                # Check if the response is a bar chart.
                if "bar" in response_dict:
                    data = response_dict["bar"]
                    df = pd.DataFrame(data)
                    df.set_index("columns", inplace=True)
                    st.bar_chart(df)

                # Check if the response is a line chart.
                if "line" in response_dict:
                    data = response_dict["line"]
                    df = pd.DataFrame(data)
                    df.set_index("columns", inplace=True)
                    st.line_chart(df)

                # Check if the response is a table.
                if "table" in response_dict:
                    data = response_dict["table"]
                    df = pd.DataFrame(data["data"], columns=data["columns"])
                    st.table(df)


# versão que precisa mandar sempre o dataset
def analisar_dados(data_frame: pd.DataFrame, query: str):
    with st.spinner('Processando...'):
        csv_string = data_frame.to_csv(index=False)

        base_url = os.getenv('SERVER_INDEX_HTTP')
        rota = "/api/analisar_dados"
        url = f"{base_url}{rota}"

        AnalyzeDataRequest = {
            'data': csv_string,
            'query': query
        } 

        token = st.session_state.auth_token
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.post(url, headers=headers, json=AnalyzeDataRequest, timeout=360)
    
        if response.status_code == 200:
            result = response.json()
            st.write(result)


def preparar_dataset(data_frame: pd.DataFrame):
    with st.spinner('Processando...'):

        # Gerar JSON com os dados
        data_json = data_frame.to_json(orient="records")

        # Gerar JSON com os metadados (tipos de colunas)
        metadata = {col: str(data_frame[col].dtype) for col in data_frame.columns}

        AUTH0_M2M_CLIENT_ID = os.getenv('AUTH0_M2M_CLIENT_ID')

        PrepareDatasetRequest = {
            'client_id': AUTH0_M2M_CLIENT_ID,
            'data': data_json,
            'metadata': json.dumps(metadata)
        }
        
        base_url = os.getenv('SERVER_INDEX_HTTP')
        rota = "/api/preparar_dataset"
        url = f"{base_url}{rota}"

        token = st.session_state.auth_token
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.post(url, headers=headers, json=PrepareDatasetRequest, timeout=360)

        if response.status_code == 200:
            result = response.json()
            dataset_id = result['dataset_id']
            st.session_state.dataset_id = dataset_id
            st.session_state.analise = None
        else:
            st.session_state.dataset_id = None
            st.session_state.analise = None


# versão que manda apenas o dataset_id
def analisar_dataset(query: str):
    with st.spinner('Processando...'):
        base_url = os.getenv('SERVER_INDEX_HTTP')
        rota = "/api/analisar_dataset"
        url = f"{base_url}{rota}"

        token = st.session_state.auth_token
        headers = {"Authorization": f"Bearer {token}"}
        params = {"id": st.session_state.dataset_id, "query": query}
        response = httpx.get(url, headers=headers, params=params, timeout=360)
        
        result = None
        if response.status_code == 200:
            result = response.json()

        return result
