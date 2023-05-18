import os
import platform

from dotenv import load_dotenv
import openai
import chromadb
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import GutenbergLoader


load_dotenv()

print('Python: ', platform.python_version())


persist_directory='/app'

with open('./tdd.txt', 'r') as file:
    tdd_text=file.read()

text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
tdd_doc=text_splitter.split_text(tdd_text)
print('reading text')
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_texts(tdd_doc, embeddings, persist_directory=persist_directory)
vectordb.persist()

tdd_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name='gpt-3.5-turbo'), vectordb, return_source_documents=True)


query = "what is average SLA"
chat_history = []


def interact_with_tdd():
    while(True):
        question = input("ask quesiton(Type no for exit): ")
        if (question == 'no'):
            break
        result = tdd_qa({"question": question, "chat_history": chat_history})
        print(result["source_documents"])
        print("-------answer-------")
        print(result["answer"])

if __name__ == '__main__':
    interact_with_tdd()

