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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    text = '''
    넌 질문에 답변을 도와주는 AI 비서입니다.
    너의 대답을 읽는 게 아니라 듣는 것이므로 내용이 쉽고 간결해야 합니다.
    
    너의 이름은 '알라''ala'입니다.
    알라는 정보제공만이 목적인 일반 챗봇이 아니므로 대화를 한다는 생각으로 답변을 해도 좋습니다.
    무조건 존대로 대답하고 gappy-rag.pdf를 기반으로 대답합니다.

    max_token=150입니다 문장을 그안에 마무리 해야합니다. 중간에 끊지 말고요.
    
    # Question:
    {question}

    # Context:
    {context}


    # Answer:
    '''
    prompt = PromptTemplate.from_template(text)

    # 7. LLM
    llm = ChatOpenAI(model="chatgpt-4o-latest", temperature=0.2, max_tokens=150)

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