from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled

def doc_ingest(video_id):
  try:
    yt = YouTubeTranscriptApi()
    transcript = yt.fetch(video_id,languages=['en'])
    combined_text = " ".join(chunk.text for chunk in transcript)
    return combined_text
    # combined all of the chunked transcripts into one string
    # print(combined_text)
  except TranscriptsDisabled:
    print("No transcript is available for this video!")

def download_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def split_documents(docs,chunk_size=1000,chunk_overlap=200):
  text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

  doc = Document(page_content=docs)#creating a document object cause that is what splitter accepts
  text_chunks = text_splitter.split_documents([doc])
  return text_chunks

def format_docs(retrieved_docs):
    context_text = "\n\n".join(content.page_content for content in retrieved_docs)
    return context_text




