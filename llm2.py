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


def llm_rag2(user_input):
    # loader = PyMuPDFLoader()
    # docs = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # split_docs = text_splitter.split_documents(docs)
    # embeddings = OpenAIEmbeddings()
    # vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    # retriever = vectorstore.as_retriever()

    text = '''
다음은 특정 사용자가 읽은 뉴스 카테고리 데이터를 기반으로 한 요약입니다. 
제공된 데이터는 감정, 대분야, 소분야, 유형의 순서로 정리되어 있습니다. 
감정은 기사의 논조가 긍정적인지 부정적인지 균형적인지에 관한 것입니다.
대분야는 경제와 사회 문화라는 큰 카테고리이고 소분야는 그에 속하는 분야입니다.
유형은 대분야와 소분야를 토대로 관심사를 유형화한 것.
이를 바탕으로 사용자를 분석/요약하는 내용을 작성할 것. 

아래 형식을 반드시 준수하고, 분석적인 어조로 작성하며, 이모티콘을 추가하여 6줄 이내로 설명해 주세요.

**출력 형식**
Address: ECBS from Seoul, South Korea
Date: [2024-11]

Description----------------


[소분야]를 선택한 당신은 [유형]입니다! 
[대분야] 분야 중 [소분야]에 관심이 많습니다! 😊💡
 [추가적인 특징 1] 🌱 
 [추가적인 특징 2] 📈

**주의사항:**

**Description 섹션:**
   - 첫 문장은 고정된 형식을 사용하세요: [소분야]를 선택한 당신은 [유형]! ...
   - 이후 문장은 자연스럽게 이어지도록 작성하되, 필수 요소를 포함하세요.
   

 **형식 준수:**
   - 대괄호([])는 실제 내용으로 작성.
   - 본문은 120token이하로 작성하려 노력할 것
   - 마지막 문장은 "당신을 위한 추천 기사를 스크롤을 내려서 확인해 보세요!"로 고정할 것.


    # Question:
    {question}

    # Answer:
    '''
    prompt = PromptTemplate.from_template(text)

    # 7. LLM
    llm2 = ChatOpenAI(model="chatgpt-4o-latest", temperature=0.2, max_tokens=200)
    

    parser = StrOutputParser()
    # 8. Chains
    chain = (
        {'question': RunnablePassthrough()}
        | prompt
        | llm2
        | parser
    )

    ans1 = chain.invoke(user_input)
    return ans1