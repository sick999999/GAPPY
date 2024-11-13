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
유형은 대분야와 소분야를 토대로 관심사를 유형화한 것입니다.
이를 바탕으로 사용자를 간단히 요약하는 내용을 작성해 주세요. 
Date에는 현재 날짜를 yyyy-mm-DD 형식으로 작성해주세요. 

아래 형식을 반드시 준수하고, 친근하고 분석적인 어조로 작성하며, 이모티콘을 추가하여 8줄 이내로 설명해 주세요.

**출력 형식**
Address: South Korea, 
Date: [yymmdd],

Description---------------------------------


[소분야]를 선택한 [유형]! 낭만적인 연애를 꿈꾸고 있군요!
사랑하는 사람과 낭만적인 일이 계속되면, 기분이 좋아져서 다른 일도 덩달아 열심히 하는 타입이시군요! 당신은 [대분야] 분야 중 [소분야]를 자주 읽는 사람입니다! 😊💡
 [추가적인 특징 1] 🌱 
 [추가적인 특징 2] 📈

**주의사항:**

1. **점선:**
   - 시작과 끝에 점선 두 줄을 사용하세요.
   - Address와 Date는 오른쪽 정렬로 배치하고, 각 항목은 별도의 줄에 표시하세요.
   - Description 다음에는 빈 줄을 추가하세요.
   - TOTAL은 오른쪽 정렬로 표시하세요.
   - Date에 날짜는 반드시 당일 해당 년도와 월일을 적으세요(yy년 mm월 dd일)

2. **Description 섹션:**
   - 첫 문장은 고정된 형식을 사용하세요: [소분야]를 선택한 [유형]! ...
   - 이후 문장은 자연스럽게 이어지도록 작성하되, 필수 요소를 포함하세요.
   - 이모티콘을 적절히 사용하여 친근한 분위기를 조성하세요.
   - 추가적인 특징들은 이모티콘과 함께 간략하게 설명하세요.

3. **형식 준수:**
   - 모든 대괄호([])는 실제 내용으로 대체하세요.
   - 문단과 문장 사이에 적절한 공백을 유지하여 가독성을 높이세요.
   - 전체 요약은 8줄 이내로 유지하세요.

**출력 예시**
Address: South Korea
Date: mm/dd/yy
Description

생활경제를 선택한 알뜰 살림꾼! 절약을 잘하고 자산을 효율적으로 관리하는 당신은 경제 분야 중 생활경제를 자주 읽는 사람입니다! 😊

💡 재테크와 투자에 관심이 많아 재정 안정을 위해 노력하는 모습이 인상적이에요. 
👨‍👩‍👧‍👦 가족의 경제적 안정을 위해 항상 계획을 세우고, 친환경 소비에도 신경을 쓰는 친절한 분이시군요! 
🌱 📈 금융 관련 뉴스와 정보를 꾸준히 찾아보며, 자신의 자산을 효과적으로 늘려가고 있어요.
    


    # Question:
    {question}

    # Answer:
    '''
    prompt = PromptTemplate.from_template(text)

    # 7. LLM
    llm2 = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    

    parser = StrOutputParser()
    # 8. Chain
    chain = (
        {'question': RunnablePassthrough()}
        | prompt
        | llm2
        | parser
    )

    ans1 = chain.invoke(user_input)
    return ans1