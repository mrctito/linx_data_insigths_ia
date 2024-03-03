from typing import Dict, Any, List, Optional, Union
import os
import json
import langchain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import (LLMChain, ConversationalRetrievalChain, ConversationChain, RetrievalQA, RetrievalQAWithSourcesChain, QAGenerationChain)
from langchain.schema import BaseRetriever, Document, LLMResult
from langchain.prompts import PromptTemplate
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter


AVISO_SERVICO_ANALISE_INDICADORES = '''
Gostaríamos de informar que o serviço que você utilizou recentemente é uma demonstração de nossa capacidade tecnológica e operacional. É importante destacar que, se você enviou mais de uma planilha para análise, somente a primeira foi considerada nesta demonstração. Essa limitação é específica para o nosso serviço de demonstração e não reflete as capacidades plenas de nossos serviços regulares.

Queremos também apresentar a flexibilidade deste serviço. Ele foi desenvolvido para ser facilmente integrado em diferentes cenários e plataformas. Aqui estão algumas das opções de integração disponíveis:

Acionamento por E-mail: Você pode acionar nosso serviço simplesmente enviando um e-mail com uma planilha anexada. Isso é ideal para situações em que a rapidez e a conveniência são essenciais.

Integração via API: Nossa API robusta permite uma integração direta com o seu produto ou sistema. Isso proporciona uma solução mais automatizada e integrada, ideal para aplicativos ou plataformas digitais que requerem análises de imagem frequentes.

Interface Específica: Oferecemos também uma interface dedicada para o uso deste serviço. Essa opção é perfeita para usuários que preferem uma solução mais visual e interativa.

Além disso, nosso serviço oferece a capacidade de realizar comparações com análises anteriores. Isso significa que você pode observar mudanças e desenvolvimentos ao longo do tempo, o que é particularmente útil para monitorar progressos ou identificar tendências.

Esperamos que esta demonstração tenha sido esclarecedora e útil. Estamos à disposição para discutir como nosso serviço pode ser adaptado às suas necessidades específicas e como podemos ajudar a integrá-lo de forma mais eficaz no seu negócio.
'''

async def api_analisar_indicadores(nome_base_conhecimento:str, texto_indicadores: str, texto_valores: str, texto_prompt: str=None):
    print("Analisando indicadores...")

    if not nome_base_conhecimento:
        return "Base de conhecimento precisa ser informada."

    if not texto_indicadores:
        return "Tabela para análise precisa ser informada."
    
    if not texto_valores:
        return "Valores para análise precisam ser informados."
    
    debug = False
    session = create_db_session_admin()
    try:
        base_conhecimento = session.query(BaseConhecimento).filter_by(nome_base=nome_base_conhecimento).first()
        if not base_conhecimento:
            print("Base de conhecimento não encontrada.")
            return "Base de conhecimento não encontrada." 
        debug = base_conhecimento.debug >= 2        
    finally:
        session.close()
    
    instrucoes_default =  '''
    -Faça uma análise individualizada do indicador, considerando seus valores de referência mínimo, máximo, ideal e inter-relações com os demais indicadores.
    -Faça uma analise holistica minuciosa do indicador sob a ótica da coluna "inter_relacionamento" de cada um dos indicadores. Explique detalhadamente os impactos entre os indicadores que se inter-relacionam.
    -Escreva as ações corretivas necessárias para alcançar os valores ideais do indicador, determinando a prioridade de execução.
    -Avalie os impactos na saúde financeira da empresa e explique detalhadamente.    
    -Avalie os impactos na operação da empresa e explique detalhadamente.    
    -Identifique riscos potenciais associados ao indicador e oportunidades que podem ser exploradas e explique detalhadamente.
    -Escreva sugestões práticas de ações para alcançar os valores ideais do indicador. Especifique a ordem de prioridade das ações e o impacto esperado. Sugira as ações para mitigar os riscos e explorar as oportunidades identificadas na análise.
    -Sugira indicadores adicionais que deveriam ser incluídos na tabela para melhorar a análise holistica.
    -LEMBRE-SE: você NÃO deverá fazer a análise apenas do primeiro indicador, mas sim de todos os indicadores. É imprescindível que você faça a análise de todos os indicadores individualmente e em conjunto, levando em consideração a coluna inter_relacionamento.
    -PRESTE ATENÇÃO: Escreva um relatório muito completo, organizado, bem redigido e minucioso que será apresentado para a diretoria da companhia.
    '''
    
    if texto_prompt == None:
        texto_prompt = instrucoes_default
    
    prompt_template = '''
    Você é especialista em análise de indicadores de negócios e foi contratado para analisar os indicadores de uma empresa e produzir um relatório completo e minucioso para a diretoria da companhia.
    
    Vou lhe passar duas tabelas em formato JSON:
    
      1- Uma tabela com a lista de até 20 indicadores de negócios a serem analisados.
      2- A outra tablela com os valores atuais desses indicadores.
    
    Você deverá combinar essas duas tabelas pelo campo "codigo_indicador", analisar os indicadores conforme instruções abaixo e escrever um relatório completo e minucioso com as suas conclusões.
    Você deverá analisar os indicadores individualmente, mas também deverá analisar os indicadores em conjunto, aplicando uma visão holistica em como eles se inter-relacionam e como impactam os processos e resultados da conmpanhia.
    A coluna inter_relacionamento é a chave para você analisar os indicadores em conjunto.
    Siga exatamente as instruções abaixo. Não invente informações. Faça análises completas e minuciosas. Não deixe de analisar nenhum indicador.
    
    INSTRUÇÕES PARA ANÁLISE DOS INDICADORES:
    "{texto_prompt}"
    
    TABELA DE INDICADORES A SEREM ANALISADOS:
    "{texto_indicadores}"
        
    TABELA DE VALORES ATUAIS DOS INDICADORES:
    "{texto_valores}"

    ANÁLISE:
    ESCREVA AQUI A ANALISE COMPLETA CONSIDERANDO TODOS OS INDICADORES.
    '''
    
    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(temperature=0, model_name=base_conhecimento.openai_model, openai_api_key=base_conhecimento.openai_api_key, verbose=debug)
    
    chain = LLMChain(llm=llm, prompt=prompt, verbose=debug)
    result_text = await chain.arun(texto_indicadores=texto_indicadores, texto_valores=texto_valores, texto_prompt=texto_prompt)

    if hasattr(result_text, 'transformed_content'):
        texto_extraido = result_text.transformed_content
    elif hasattr(result_text, 'raw_content'):
        texto_extraido = result_text.raw_content
        return texto_extraido
    else:
        texto_extraido = result_text
    
    return texto_extraido

