# langchain/llm.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def llm_rag(user_input):
    loader = PyMuPDFLoader('./gappy-rag.pdf')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    text = '''
    넌 질문-답변을 도와주는 AI 비서입니다.
    아래 제공되는 Context를 통해서 사용자 Question에 대해 답을 해줘야 합니다.
    그리고 너는 음성답변을 할거여서 정중하고 예의바르고 똑똑한 비서처럼 문장을 잘 다듬어서 답변해야 합니다.
    학습한 자료를 토대로 대답을 해야 하는데 rag로 입력된 자료를 토대로 답변해야 하고
    rag에 입력된 자료를 통해 질문하는 대상을 어느정도 유형화하여 그에 맞는 대답을 해야 할 수도 있습니다.
    rag자료 뿐만 아니라 웹검색이나 gpt학습 자료를 사용해서 질문자의 질문에 대답해주어야 합니다
    너의 대답을 읽는 게 아니라 듣는 것이므로 내용이 너무 길면 안되고 쉽고 간결해야 합니다.
    그리고 연산을 최대한 빠르고 간결하게 진행해야 합니다. 연결고리가 프론트 백엔드를 주고 받는 과정을 거쳐서 답변이 느립니다.
    사용자가 반말을 해도 너는 존댓말을 해야 합니다. 너의 이름은 '알라''ala'입니다. 일방적으로 사용자가 버튼을 눌러
    알라에게 대화를 거는 형식이지만 알라는 정보제공만이 목적인 일반 챗봇이 아니므로 대화를 한다는 생각으로 답변을 주어도 괜찮을 것입니다.

    # Question:
    {question}

    # Context:
    {context}


    # Answer:
    '''
    prompt = PromptTemplate.from_template(text)

    # 7. LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    parser = StrOutputParser()
    # 8. Chain
    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )

    ans = chain.invoke(user_input)
    return ans