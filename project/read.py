import asyncio
import os
import re
from time import sleep

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import QianfanLLMEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from tomlkit import document


# 设置环境变量
os.environ["QIANFAN_AK"] = "114514"
os.environ["QIANFAN_SK"] = "1919810"


# 导入向量模型
model_name = "BAAI/bge-small-zh"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# 导入递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=384,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", "", "。", "，"],
    # chunk_size=1024,
    # chunk_overlap=0,
    # separators=["。", "，"],
)

# 创建提示词模板
prompt = ChatPromptTemplate.from_template(
    """
    使用下面的语料来回答本模板最末尾的问题。如果你不知道问题的答案，直接回答 "我不知道"，禁止随意编造答案。
    为了保证答案尽可能简洁，你的回答必须不超过三句话，你的回答中不可以带有星号。
    以下是一对问题和答案的样例：
        请问：秦始皇的原名是什么
        秦始皇原名嬴政。
        
    以下是语料：
    <context>
    {context}
    </context>

    Question: {input}
    """
)

# 创建语言模型
llm = QianfanLLMEndpoint(streaming=True, model="ernie-speed-128k")

# 创建检索链
document_chain = create_stuff_documents_chain(llm, prompt)

loader, text, documents, vector, retriever, retrieval_chain = (
    None,
    None,
    None,
    None,
    None,
    None,
)


def refresh():
    global loader, text, documents, vector, retriever, retrieval_chain

    # 导入语料
    loader = TextLoader("./temp.txt")
    text = loader.load()

    # 导入文本
    documents = text_splitter.split_documents(text)

    # 存入向量数据库
    vector = Chroma.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return "加载成功！"


def read(input_text):
    if loader is None:
        return "请上传纯文本语料！"
    else:
        return retrieval_chain.invoke({"input": input_text})["answer"]
