# test_RAG
# 🌸 Tokbom RAG Chatbot

문서 기반 질문 응답을 위한 **RAG (Retrieval-Augmented Generation)** 챗봇입니다.  
OpenAI의 ChatGPT 또는 최적화된 Open LLaMA 모델을 활용하여, 업로드된 문서를 기반으로 정확하고 유용한 답변을 생성합니다.

![Tokbom UI](https://your-image-url.com/screenshot.png) <!-- 필요 시 대체 -->

---

## 🔍 주요 기능

- 📄 **PDF / TXT 문서 업로드 및 분할 처리**
- 💾 **FAISS 또는 Chroma 기반 벡터 DB 구축**
- 🧠 **OpenAI ChatGPT 또는 로컬 Open LLaMA 사용 가능**
- ⚙️ **4bit/8bit 양자화 및 Flash Attention 옵션**
- 💬 **대화기록 기반 RAG 응답 생성**
- 🖥️ **Streamlit 기반 웹 UI**
- 📊 **GPU 메모리 모니터링, 응답속도 확인**

---

## 🚀 데모 실행 방법

### 1. 환경 세팅

```bash
# Python 가상환경 권장
python -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
