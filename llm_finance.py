from email.encoders import encode_noop
import ofxparse
import pandas as pd
import os
from datetime import datetime

df = pd.DataFrame()
for extrato in os.listdir('extratos'):
  with open(f'extratos/{extrato}', encoding='ISO-8859-1') as ofx_file:
    ofx = ofxparse.OfxParser.parse(ofx_file)
    transactions_data = []
    
  for account in ofx.accounts: # type: ignore
    for transaction in account.statement.transactions:
      transactions_data.append({
        "Data":transaction.date,
        "Valor":transaction.amount,
        "Descrição":transaction.memo,
        "ID":transaction.id,
        "Categoria":None
      })
      
  df_temp = pd.DataFrame(transactions_data)
  df_temp["Valor"] = df_temp["Valor"].astype(dtype=float)
  df_temp["Data"] = df_temp["Data"].apply(func=lambda x: x.date())
  df = pd.concat([df, df_temp])
print(df)  

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers.string import StrOutputParser
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

template = """
Você é um analista de dados, trabalhando em um projeto de limpeza de dados.
Seu trabalho é escolher uma categoria adequada para cada lançamento financeiro que vou te
enviar.

Todos são transações financeiras de uma pessoa física.

Escolha uma dentre as seguintes categorias:
- Alimentação
- Receitas
- Saúde
- Mercado
- Educação
- Compras
- Transporte
- Investimento
- Transferências para terceiros
- Telefone
- Moradia

Escolha a categoria deste item:
{text}

Responda apenas com a categoria.
"""

prompt = PromptTemplate.from_template(template=template)
chat = ChatGroq(model="llama-3.1-70b-versatile") # type: ignore
chain = prompt | chat

category = []
for transaction in list(df["Descrição"].values):
  category += [chain.invoke(input=transaction).content] # type: ignore
df["Categoria"] = category
print(df)