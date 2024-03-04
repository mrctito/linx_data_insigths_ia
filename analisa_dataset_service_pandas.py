from io import StringIO, BytesIO
from typing import Any, Dict, List, Optional
from numpy import double, require
import jwt
import typing
from pydantic import BaseModel
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits.pandas.prompt import PREFIX
from langchain.prompts import PromptTemplate
from llm import cria_llm



def ajustar_tipos_colunas(df: pd.DataFrame):


    for column in df.columns:
        col_type = df[column].dtype  # Obtem o tipo de dados da coluna atual
        
        # Verifica e ajusta os tipos de dados conforme necessário
        if pd.api.types.is_datetime64_any_dtype(col_type):
            df[column] = pd.to_datetime(df[column])
        elif pd.api.types.is_float_dtype(col_type):
            df[column] = df[column].astype(float)
        elif pd.api.types.is_integer_dtype(col_type):
            df[column] = df[column].astype(int)
        elif pd.api.types.is_bool_dtype(col_type):
            df[column] = df[column].astype(bool)
        elif pd.api.types.is_string_dtype(col_type) or pd.api.types.is_object_dtype(col_type):
            df[column] = df[column].astype(str)


async def svc_analisar_dataset_pandas(json_obj: Dict[str, Any], query: str, verbose: bool = False):
    print("Analisando dataset via pandas...")
    prompt = (
        """
            Siga as instruções Para a consulta a seguir:
            
            Responda sempre em Português.

            Você trabalhará sempre com os dados em formato Pandas Dataframe.

            Sempre dê a resposta correta em formato de texto da seguinte forma:
            {"answer": "answer"}
            Exemplo:
            {"answer": "O título com a classificação mais alta é 'Gilead'"}

            Retorne toda a saída como uma string.

            Pense sempre passo a passo.

            Abaixo está a consulta.

            Query: 
            """
        + query
    )

    prompt = PromptTemplate.from_template(prompt)

    resultado = ""
    df = pd.DataFrame(json_obj)
    ajustar_tipos_colunas(df)
    llm = cria_llm(prompt)

    prefix = "Responda sempre em Português!\n"+PREFIX
    agent = create_pandas_dataframe_agent(llm, df, prefix=prefix, max_iterations=60, max_execution_time=120, handle_parsing_errors=True, verbose=verbose, agent_executor_kwargs={"handle_parsing_errors": True})
    resultado = agent.run(prompt) 
    return resultado 