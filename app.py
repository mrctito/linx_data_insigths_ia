import os
import json
import asyncio
import aioconsole
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn
from analisa_dataset_client import executar_analise_dataset



def main():
  executar_analise_dataset()


if __name__ == "__main__":
  main()