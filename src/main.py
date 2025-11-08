from .utils import doc_ingest,split_documents,download_embeddings,format_docs
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

video_id = "TLQmMsmlRo8"
combined_text = doc_ingest(video_id)
#doc_ingest combines the splitted chunks of yt transcript and ouputs as a single string ; 

split_chunks = split_documents(combined_text,500,50)
#split_documents(document,chunk_size,chunk_overlap) : splits the entire doc into chunks

embeddings = download_embeddings()

vector_store = FAISS.from_documents(split_chunks,embeddings)
#store those chunks as embeddings 

retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})
#retrieves the document from the vector store on the basis of semantics similarity

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key


llm = GoogleGenerativeAI(model="gemini-2.5-flash")
#intiating the model;


prompt_template = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context of the video.
    If the context is insufficient, just say that you donot know the answer.
    Context: {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)


#building a chain -- a pipeline so that none of the processes be manually invoked and the orchestration be handled by the pipeline itself ; 


parallel_chain = RunnableParallel({
    "context":retriever | RunnableLambda(format_docs),
    "question":RunnablePassthrough()
}
)
# a parallel chain where :
#retriever gets the question -- input -- and feeds it into format_docs (converted into runnable through runnablelambda)

parser = StrOutputParser()
main_chain = parallel_chain | prompt_template | llm | parser
#parallel chain combined with prompt and llm to parse the ouptut ultimately forming a linear chain !

