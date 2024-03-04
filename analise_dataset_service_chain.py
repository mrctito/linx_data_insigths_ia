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
            Você receberá uma tabela em formato Json e deverá responder a pergunta do usuário com base nela.

            Siga as instruções abaixo para realizar a tarefa:
            
              -Responda sempre em Português.
              -Não inclua a tabela na resposta.
              -Não inclua seu raciocínio na resposta.
              -Retorne a resposta como uma string.

            Aqui está a tabela de dados em formato JSON:
            {tabela}

            Aqui está a pergunta do usuário:
            {query}
            """
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["tabela", "query"]
    )

    chain = cria_chain(prompt, verbose=True)
    result_text = chain.invoke({"tabela": tabela_json_str, "query": query})

    if hasattr(result_text, 'text'):
        texto_extraido = result_text["text"]
    else:
        texto_extraido = result_text
    
    return texto_extraido