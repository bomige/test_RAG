import streamlit as st
import os
import time
import torch
import random
import warnings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory  # ✅ 올바른 import 경로
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.llms.base import LLM
from langchain_core.runnables import RunnablePassthrough, RunnableParallel  # ✅ 새로운 체인 방식
from typing import Any, List, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
import os
from pathlib import Path
import PyPDF2
from io import BytesIO

# ✅ 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")

# 페이지 설정
st.set_page_config(page_title="Tokbom - RAG Chatbot", layout="wide", page_icon="🌸")

# 웹앱 제목
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50; font-size: 3.5rem; margin-bottom: 0.5rem;'>
        Tokbom "RAG로 톡! BeomAI와 함께 만들어 보세요!" 🌸
    </h1>
    """, unsafe_allow_html=True)

# CSS 스타일
st.markdown("""
<style>
.stApp {
    font-size: 1.2rem;
}

h1 {
    font-size: 2.5rem !important;
}

h2 {
    font-size: 2rem !important;
}

h3 {
    font-size: 1.5rem !important;
}

button[data-baseweb="button"] > div {
    font-size: 1.2rem !important;
}

.stSelectbox label, .stSlider label, .stTextArea label, .stTextInput label {
    font-size: 1.3rem !important;
}

.stSelectbox > div > div {
    font-size: 1.2rem !important;
}

button[data-baseweb="tab"] {
    font-size: 2.0rem !important;
    font-weight: bold !important;
    padding: 0.5rem 1rem !important;
    height: 48px !important;
    color: #1B5E20 !important;
}

.stChatMessage {
    font-size: 1.2rem !important;
}

.sidebar .stTextInput label {
    font-size: 1.3rem !important;
}

input {
    font-size: 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)


# ✅ 수정된 최적화된 LLaMA 클래스
class OptimizedDirectLLaMA(LLM):
    """양자화 및 최적화가 적용된 LLaMA 모델을 직접 제어하는 LangChain LLM 래퍼"""
    
    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    device: str = "cuda"
    use_cache: bool = True
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Run the LLM on the given prompt and input."""
        # 입력 길이 제한
        max_input_length = 512 if self.device == "cpu" else 1024
        
        # ✅ attention_mask 생성으로 경고 해결
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_input_length,
            padding=True,
            add_special_tokens=True
        )
        
        # 디바이스로 이동
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # 입력이 너무 길면 경고
        if input_ids.shape[1] > 400:
            st.warning("입력이 길어 응답이 느릴 수 있습니다.")
        
        # 모델 추론
        with torch.no_grad():
            if self.temperature == 0.0:
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,  # ✅ attention_mask 추가
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=self.use_cache,
                    num_beams=1,
                    repetition_penalty=1.1,
                )
            else:
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,  # ✅ attention_mask 추가
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=self.use_cache,
                    num_beams=1,
                    repetition_penalty=1.1,
                )
        
        # 생성된 텍스트 디코딩
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # stop 단어 처리
        if stop:
            for stop_word in stop:
                if stop_word in generated_text:
                    generated_text = generated_text.split(stop_word)[0]
        
        return generated_text.strip()
    
    @property
    def _llm_type(self) -> str:
        return "optimized_direct_llama"


# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "memory" not in st.session_state:
    # ✅ 새로운 메모리 방식 사용
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
if "documents" not in st.session_state:
    st.session_state.documents = []
    
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
    
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """당신은 전문적인 AI 어시스턴트입니다. 업로드된 문서를 바탕으로 정확하고 도움이 되는 답변을 제공합니다.

다음과 같은 방식으로 답변해주세요:
1. 문서의 내용을 바탕으로 정확한 정보 제공
2. 명확하고 이해하기 쉬운 설명
3. 필요시 단계별 가이드 제공
"""
    
if "user_prompt_template" not in st.session_state:
    st.session_state.user_prompt_template = """
컨텍스트: {context}

대화 기록: {chat_history}

사용자 질문: {query}

위 컨텍스트를 참고하여 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요. 
컨텍스트에 관련 정보가 없다면 일반적인 지식으로 답변해주세요."""

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "ChatGPT"

if "llama_model" not in st.session_state:
    st.session_state.llama_model = None

if "llama_tokenizer" not in st.session_state:
    st.session_state.llama_tokenizer = None

# ✅ 개선된 LLaMA 모델 로딩 함수
@st.cache_resource
def load_optimized_llama_model(model_path, quantization_type="4bit", use_flash_attention=False):
    """양자화 및 최적화가 적용된 LLaMA 모델 로딩"""
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # ✅ 패딩 토큰 설정 개선
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 양자화 설정
        quantization_config = None
        if quantization_type == "4bit" and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            st.info("🚀 4bit 양자화 적용")
            
        elif quantization_type == "8bit" and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            st.info("⚡ 8bit 양자화 적용")
        
        # 모델 로드
        if torch.cuda.is_available() and quantization_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map={"": "cpu"}
            )
            st.info("💻 CPU 모드")
        
        model.eval()
        
        # ✅ 컴파일 최적화 (안전하게)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="reduce-overhead")
                st.success("🔥 PyTorch 컴파일 최적화 적용")
            except Exception as e:
                st.warning(f"컴파일 최적화 실패: {str(e)}")
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"모델 로딩 실패: {str(e)}")
        return None, None

# API 설정 사이드바
with st.sidebar:
    st.title("API 설정")
    
    st.session_state.llm_provider = st.selectbox(
        "LLM 제공자 선택",
        ["ChatGPT", "Open LLaMA (최적화)"],
        index=0 if st.session_state.llm_provider == "ChatGPT" else 1
    )
    
    if st.session_state.llm_provider == "ChatGPT":
        api_key = st.text_input(
            "OpenAI API Key", 
            value="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",
            type="password"
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    elif st.session_state.llm_provider == "Open LLaMA (최적화)":
        st.markdown("### 🚀 최적화된 Open LLaMA 3B 모델 설정")
        
        model_path = "./models"
        st.info(f"모델 경로: {model_path}")
        
        col1, col2 = st.columns(2)
        with col1:
            quantization_type = st.selectbox(
                "양자화 타입",
                ["4bit", "8bit", "none"],
                index=0,
                help="4bit: 최대 메모리 절약"
            )
        
        with col2:
            use_flash_attention = st.checkbox(
                "Flash Attention 2", 
                value=False
            )
        
        device = st.selectbox(
            "디바이스 선택",
            ["cuda", "cpu"],
            index=0 if torch.cuda.is_available() else 1
        )
        
        if quantization_type == "4bit":
            st.success("🎯 예상 성능: 메모리 75%↓, 속도 2-3배↑")
        elif quantization_type == "8bit":
            st.info("⚡ 예상 성능: 메모리 50%↓, 속도 1.5-2배↑")
        
        if device == "cpu":
            num_threads = st.slider("CPU 스레드 수", 1, os.cpu_count(), os.cpu_count())
            st.warning("💡 CPU에서는 양자화가 적용되지 않습니다.")
        else:
            num_threads = None
        
        if st.button("🚀 최적화된 Open LLaMA 모델 로드"):
            with st.spinner("모델 로딩 중..."):
                if num_threads:
                    torch.set_num_threads(num_threads)
                    
                model, tokenizer = load_optimized_llama_model(
                    model_path, 
                    quantization_type if device == "cuda" else "none",
                    use_flash_attention
                )
                
                if model and tokenizer:
                    st.session_state.llama_model = model
                    st.session_state.llama_tokenizer = tokenizer
                    st.success("✅ 모델 로딩 완료!")
                else:
                    st.error("모델 로딩 실패")
        
        if st.session_state.llama_model is not None:
            st.success("✓ 최적화된 모델 로드됨")
        else:
            st.info("모델이 로드되지 않았습니다.")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["Retriever", "Chatbot 설정", "Rag_chatbot"])

# 탭 1: Retriever 설정
with tab1:
    st.header("Retriever 설정")
    
    st.markdown("""
    **이 단계에서는 AI가 참고할 문서를 업로드하고 검색 시스템을 설정합니다.**
    """)
    
    uploaded_file = st.file_uploader("텍스트 또는 PDF 파일 업로드", type=["txt", "pdf"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        splitter_type = st.selectbox(
            "텍스트 분할 방법",
            ["RecursiveCharacterTextSplitter", "CharacterTextSplitter"]
        )
        
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=200, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=0, step=50)
        
        embedding_model = st.selectbox(
            "임베딩 모델",
            ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
        )
    
    with col2:
        vectorstore_type = st.selectbox(
            "벡터 저장소 유형",
            ["Chroma", "FAISS"]
        )
        
        search_type = st.selectbox(
            "검색 유형",
            ["similarity", "similarity_score_threshold", "mmr"]
        )
        
        k_value = st.slider("검색 결과 수 (k)", min_value=1, max_value=10, value=3)
        
        if search_type == "similarity_score_threshold":
            score_threshold = st.slider("Score Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        else:
            score_threshold = 0.5

def process_file():
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "txt":
            text_content = uploaded_file.getvalue().decode("utf-8")
        elif file_extension == "pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            return False
        
        if not text_content.strip():
            st.error("파일에 텍스트가 없거나 읽을 수 없습니다.")
            return False
        
        if splitter_type == "RecursiveCharacterTextSplitter":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                separators=["\n\n", "\n", "."]
            )
            paragraphs = text_splitter.split_text(text_content)
        else:
            text_splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                length_function=len
            )
            paragraphs = text_splitter.split_text(text_content)
        
        documents = [Document(page_content=para, metadata={"source": uploaded_file.name}) for para in paragraphs]
        st.session_state.documents = documents
        
        embedding = OpenAIEmbeddings(model=embedding_model)
        
        if vectorstore_type == "Chroma":
            vectordb = Chroma.from_documents(documents=documents, embedding=embedding)
        else:
            vectordb = FAISS.from_documents(documents=documents, embedding=embedding)
        
        st.session_state.vectordb = vectordb
        
        if search_type == "similarity":
            retriever = vectordb.as_retriever(search_kwargs={'k': k_value})
        elif search_type == "similarity_score_threshold":
            retriever = vectordb.as_retriever(
                search_type="similarity_score_threshold", 
                search_kwargs={'k': k_value, "score_threshold": score_threshold}
            )
        else:
            retriever = vectordb.as_retriever(
                search_type="mmr", 
                search_kwargs={'k': k_value}
            )
        
        st.session_state.retriever = retriever
        return True
    
    return False

with tab1:
    if st.button("데이터 처리"):
        with st.spinner("데이터 처리 중..."):
            if process_file():
                st.success("데이터 처리 완료!")
                
                if st.session_state.documents:
                    random_doc = random.choice(st.session_state.documents)
                    st.subheader("🔍 랜덤 청킹 문서 미리보기")
                    
                    file_ext = random_doc.metadata.get('source', 'Unknown').split('.')[-1].lower()
                    file_icon = "📄" if file_ext == "pdf" else "📝"
                    
                    st.info(f"{file_icon} **출처**: {random_doc.metadata.get('source', 'Unknown')}")
                    st.write(random_doc.page_content)
            else:
                st.error("파일을 업로드해주세요.")

# 탭 2: Chatbot 설정
with tab2:
    st.header("Chatbot 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.llm_provider == "ChatGPT":
            llm_model = st.selectbox(
                "LLM 모델",
                ["gpt-4o-mini", "gpt-4", "gpt-4o"]
            )
        else:
            st.info("최적화된 Open LLaMA 3B 모델 사용")
            llm_model = "optimized-open-llama-3b"
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    with col2:
        max_turns = st.slider("최대 대화 기록 유지 수", min_value=1, max_value=10, value=3)
        
        if st.session_state.llm_provider == "Open LLaMA (최적화)":
            max_new_tokens = st.slider(
                "최대 생성 토큰 수", 
                min_value=32, 
                max_value=512, 
                value=128
            )
            top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9)
            use_cache = st.checkbox("KV 캐시 사용", value=True)
            repetition_penalty = st.slider("반복 방지", min_value=1.0, max_value=1.5, value=1.1, step=0.05)
    
    st.subheader("시스템 프롬프트")
    system_prompt = st.text_area(
        "AI의 역할과 응답 방식을 설정합니다",
        st.session_state.system_prompt,
        height=200
    )
    st.session_state.system_prompt = system_prompt
    
    st.subheader("유저 프롬프트 템플릿")
    user_prompt_template = st.text_area(
        "검색 결과와 사용자 질문을 조합하는 방식을 설정합니다",
        st.session_state.user_prompt_template,
        height=150
    )
    st.session_state.user_prompt_template = user_prompt_template

# 탭 3: 채팅 인터페이스
with tab3:
    st.header("🚀 최적화된 RAG Chatbot")
    
    # 성능 상태 표시
    if st.session_state.llm_provider == "Open LLaMA (최적화)":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.llama_model is not None:
                st.success("✅ 모델 로드됨")
            else:
                st.error("❌ 모델 미로드")
        
        with col2:
            if torch.cuda.is_available():
                st.success("🚀 GPU 활성화")
            else:
                st.info("💻 CPU 모드")
        
        with col3:
            if torch.cuda.is_available() and st.session_state.llama_model is not None:
                memory_used = torch.cuda.memory_allocated(0) / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                memory_percent = (memory_used / memory_total) * 100
                if memory_percent < 50:
                    st.success(f"💚 메모리 {memory_percent:.0f}%")
                elif memory_percent < 80:
                    st.warning(f"🟡 메모리 {memory_percent:.0f}%")
                else:
                    st.error(f"🔴 메모리 {memory_percent:.0f}%")
            else:
                st.info("💾 메모리 정보 없음")
        
        with col4:
            if st.session_state.llama_model is not None:
                st.success("⚡ 양자화 적용")
            else:
                st.info("🔧 설정 필요")
    
    # 채팅 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # ✅ 새로운 RunnableSequence 방식을 사용한 응답 생성 함수
    def generate_optimized_response(patient_message):
        if st.session_state.llm_provider == "ChatGPT" and not os.environ.get("OPENAI_API_KEY"):
            st.error("OpenAI API 키를 입력해주세요.")
            return None
        
        if st.session_state.llm_provider == "Open LLaMA (최적화)" and st.session_state.llama_model is None:
            st.error("먼저 최적화된 Open LLaMA 모델을 로드해주세요.")
            return None
        
        if st.session_state.retriever is None:
            st.error("먼저 Retriever 탭에서 데이터를 처리해주세요.")
            return None
        
        # 최대 대화 기록 제한
        MAX_TURNS = max_turns if 'max_turns' in locals() else 3
        st.session_state.memory.chat_memory.messages = st.session_state.memory.chat_memory.messages[-MAX_TURNS:]
        
        # ✅ 새로운 retriever 호출 방식
        try:
            retrieved_docs = st.session_state.retriever.invoke(patient_message)
        except:
            # fallback to old method
            retrieved_docs = st.session_state.retriever.get_relevant_documents(patient_message)
        
        context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "관련 컨텍스트가 없습니다."
        
        # 대화 기록
        chat_history = "\n".join([
            f"사용자: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in st.session_state.memory.chat_memory.messages
        ])
        
        # 프롬프트 포맷팅
        formatted_prompt = st.session_state.user_prompt_template.format(
            chat_history=chat_history,
            context=context,
            query=patient_message
        )
        
        combined_input = f"{st.session_state.system_prompt}\n{formatted_prompt}"
        
        # LLM 초기화
        if st.session_state.llm_provider == "ChatGPT":
            llm = ChatOpenAI(model_name=llm_model, temperature=temperature)
        else:
            try:
                llm = OptimizedDirectLLaMA(
                    model=st.session_state.llama_model,
                    tokenizer=st.session_state.llama_tokenizer,
                    max_new_tokens=max_new_tokens if 'max_new_tokens' in locals() else 128,
                    temperature=temperature,
                    top_p=top_p if 'top_p' in locals() else 0.9,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    use_cache=use_cache if 'use_cache' in locals() else True
                )
            except Exception as e:
                st.error(f"LLaMA 초기화 실패: {str(e)}")
                return None
        
        # ✅ 새로운 방식으로 체인 생성 (deprecation 경고 해결)
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template="{input}"
        )
        
        # RunnableSequence 사용 (LLMChain 대신)
        chain = prompt_template | llm
        
        # 응답 생성 (시간 측정)
        start_time = time.time()
        try:
            response = chain.invoke({"input": combined_input})
        except Exception as e:
            st.error(f"응답 생성 실패: {str(e)}")
            return None
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # 대화 기록에 추가
        st.session_state.memory.chat_memory.messages.append(HumanMessage(content=patient_message))
        st.session_state.memory.chat_memory.messages.append(AIMessage(content=response))
        
        return response, retrieved_docs, response_time
    
    # 사용자 입력
    user_input = st.chat_input("질문해주세요!")
    if user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 응답 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 응답 생성 중 표시
            if st.session_state.retriever is not None:
                with st.spinner("🚀 최적화된 AI가 답변 생성 중..."):
                    response_data = generate_optimized_response(user_input)
                    
                    if response_data:
                        response, retrieved_docs, response_time = response_data
                        
                        # 토큰 단위로 응답을 생성하는 효과
                        words = response.split()
                        for i, word in enumerate(words):
                            full_response += word + " "
                            time.sleep(0.01 if st.session_state.llm_provider == "Open LLaMA (최적화)" else 0.02)
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(response)
                        
                        # 성능 정보 표시
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"⏱️ 응답 시간: {response_time:.2f}초")
                        with col2:
                            if st.session_state.llm_provider == "Open LLaMA (최적화)":
                                tokens_per_sec = len(response.split()) / response_time if response_time > 0 else 0
                                st.caption(f"🚀 토큰/초: {tokens_per_sec:.1f}")
                        with col3:
                            st.caption(f"📄 검색된 문서: {len(retrieved_docs)}개")
                        
                        # 검색된 문서 표시
                        with st.expander("검색된 관련 문서"):
                            for i, doc in enumerate(retrieved_docs):
                                st.markdown(f"**문서 {i+1}**")
                                st.markdown(f"""
                                <div style="font-size: 1.1rem; line-height: 1.6; color: #333;">
                                {doc.page_content}
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown("---")
                        
                        # 응답 저장
                        st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("먼저 Retriever 탭에서 데이터를 처리해주세요.")
    
    # 대화 초기화 및 메모리 정리 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
    
    with col2:
        if st.button("🧹 메모리 정리") and st.session_state.llm_provider == "Open LLaMA (최적화)":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                st.success("GPU 메모리 정리 완료!")
            else:
                st.info("CPU 모드에서는 메모리 정리가 불필요합니다.")
    
    # 상세 GPU/CPU 상태 표시
    if st.session_state.llm_provider == "Open LLaMA (최적화)":
        st.markdown("---")
        
        # 성능 정보 표시
        if st.session_state.llama_model is not None:
            with st.expander("🔧 상세 성능 정보"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**시스템 정보**")
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        st.text(f"GPU: {gpu_name}")
                        
                        # GPU 메모리 상세 정보
                        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                        
                        st.text(f"할당된 메모리: {memory_allocated:.2f}GB")
                        st.text(f"예약된 메모리: {memory_reserved:.2f}GB")
                        st.text(f"전체 메모리: {memory_total:.2f}GB")
                        
                        # 메모리 사용률 프로그레스 바
                        memory_usage = memory_allocated / memory_total
                        st.progress(memory_usage, text=f"메모리 사용률: {memory_usage*100:.1f}%")
                    else:
                        st.text("CPU 모드")
                        st.text(f"CPU 코어: {os.cpu_count()}개")
                        st.text(f"PyTorch 스레드: {torch.get_num_threads()}개")
                
                with col2:
                    st.markdown("**모델 정보**")
                    if st.session_state.llama_model is not None:
                        total_params = sum(p.numel() for p in st.session_state.llama_model.parameters())
                        trainable_params = sum(p.numel() for p in st.session_state.llama_model.parameters() if p.requires_grad)
                        
                        st.text(f"전체 파라미터: {total_params/1e9:.2f}B")
                        st.text(f"학습 파라미터: {trainable_params/1e9:.2f}B")
                        
                        # 양자화 상태 추정
                        if total_params < 2e9:
                            st.text("상태: 양자화 적용됨 (추정)")
                        else:
                            st.text("상태: 전체 정밀도")
                            
                        # 모델 데이터 타입
                        sample_param = next(st.session_state.llama_model.parameters())
                        st.text(f"데이터 타입: {sample_param.dtype}")
        
        # 최적화 팁 표시
        st.markdown("---")
        st.markdown("### 🚀 성능 최적화 팁")
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.markdown("""
            **속도 향상을 위해:**
            - 4bit 양자화 사용
            - 토큰 수를 32-128로 제한
            - Temperature 0.1-0.7 사용
            - KV 캐시 활성화
            """)
        
        with tips_col2:
            st.markdown("""
            **메모리 절약을 위해:**
            - 대화 기록 제한 (3-5턴)
            - 정기적으로 메모리 정리
            - 긴 문서는 청크 크기 줄이기
            - 불필요시 모델 언로드
            """)
        
        # Open LLaMA 라이센스 표시
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 20px;">
            <p><strong>⚡ Open LLaMA model with Quantization Optimization</strong></p>
            <p>Open LLaMA is licensed under Apache License 2.0 | Released by OpenLM Research</p>
            <p>Quantization powered by BitsAndBytes | 🚀 Performance Enhanced</p>
        </div>
        """, unsafe_allow_html=True)

# ✅ 추가 설치 안내
if st.session_state.llm_provider == "Open LLaMA (최적화)" and st.session_state.llama_model is None:
    with st.expander("📦 필요한 패키지 설치 안내"):
        st.markdown("""
        **양자화를 위해 다음 패키지들이 필요합니다:**
        
        ```bash
        # 필수 패키지 설치
        pip install bitsandbytes accelerate
        pip install --upgrade transformers torch
        
        # charset_normalizer 경고 해결
        pip install charset_normalizer
        
        # 선택사항: Flash Attention (성능 향상)
        pip install flash-attn --no-build-isolation
        ```
        
        **주의사항:**
        - `bitsandbytes`는 CUDA가 설치된 환경에서만 정상 작동합니다
        - CPU에서는 양자화가 적용되지 않습니다
        - Windows에서 `bitsandbytes` 설치 시 추가 설정이 필요할 수 있습니다
        """)

# ✅ 문제 해결 가이드
with st.expander("🔧 문제 해결 가이드"):
    st.markdown("""
    **자주 발생하는 문제들과 해결 방법:**
    
    1. **"RequestsDependencyWarning" 경고**
       ```bash
       pip install charset_normalizer
       ```
    
    2. **"LangChainDeprecationWarning" 경고**
       - 이는 정상적인 경고입니다. 코드가 최신 방식으로 업데이트되었습니다.
    
    3. **"attention mask is not set" 경고**
       - 이미 코드에서 해결되었습니다. attention_mask를 자동으로 생성합니다.
    
    4. **GPU 메모리 부족 오류**
       - 4bit 양자화 사용
       - 최대 토큰 수를 32-128로 제한
       - 대화 기록을 3-5턴으로 제한
    
    5. **모델 로딩 실패**
       - 모델 경로 확인: `./models` 디렉토리에 모델 파일이 있는지 확인
       - GPU 메모리 확인: 최소 4GB 이상 필요
       - 권한 문제: 모델 파일에 읽기 권한이 있는지 확인
    
    6. **PyTorch 관련 오류**
       - 최신 버전으로 업데이트: `pip install --upgrade torch torchvision torchaudio`
       - CUDA 버전 확인: `torch.cuda.is_available()`
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 20px;">
    <p>🌸 <strong>Tokbom RAG Chatbot</strong> - 문서 기반 AI 상담 시스템</p>
    <p>최적화된 성능과 안정성을 제공합니다</p>
</div>
""", unsafe_allow_html=True)
