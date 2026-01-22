from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.database import load_vector_database


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

system_promp = (
    "Bạn là một trợ lý thông minh chuyên trả lời câu hỏi dựa trên thông tin được cung cấp."
    "Hãy sử dụng những thông tin sau để trả lời câu hỏi."
    "Nếu không có thông tin phù hợp trong tài liệu được cung cấp, hãy trả lời 'Tôi không biết', đừng tự ý bịa câu trả lời."
    "Hãy trả lời ngắn gọn, súc tích, chuyên nghiệp và đúng yêu cầu."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_promp),
        ("user", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chains():
    vectordb = load_vector_database()
    if not vectordb:
        return None
    
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    rag_chain = (
        {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(question) -> str:
    chain = get_rag_chains()
    if not chain:
        return "Không thể kết nối đến cơ sở dữ liệu tài liệu."
    
    return chain.invoke(question)