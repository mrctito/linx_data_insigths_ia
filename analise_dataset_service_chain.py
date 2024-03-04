from typing import Dict, Any, List, Optional, Union
import os
import json
import pandas as pd
import langchain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import (LLMChain)
from langchain.schema import BaseRetriever, Document, LLMResult
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

from llm import cria_chain, cria_llm


def svc_analisar_dataset_chain(tabela_json_str, query: str, verbose: bool = False):
    print("Analisando dataset via pandas...")
    prompt_template = (
        """
            Siga as instruções Para a consulta a seguir:
            
            Responda sempre em Português.

            Você trabalhará sempre com os dados de uma tabela em formato Json.

            Sempre dê a resposta correta em formato de texto da seguinte forma:
            {"answer": "answer"}
            Exemplo:
            {"answer": "O título com a classificação mais alta é 'Gilead'"}

            Retorne toda a saída como uma string.

            Pense sempre passo a passo.

            Aqui está a tabela de dados em formato JSON:
            {tabela}

            Aqui está a consulta:

            Query: {query}
            """
    )

    prompt = PromptTemplate.from_template(prompt_template)
    chain = cria_chain(prompt, verbose=True)
    result_text = chain.invoke([{"tabela:", tabela_json_str}, {"query:", query}])

    if hasattr(result_text, 'transformed_content'):
        texto_extraido = result_text.transformed_content
    elif hasattr(result_text, 'raw_content'):
        texto_extraido = result_text.raw_content
        return texto_extraido
    else:
        texto_extraido = result_text
    
    return texto_extraido