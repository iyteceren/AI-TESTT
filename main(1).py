from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
# vectordb'i import et
from langchain.vectorstores import Chroma

import os
os.environ["OPENAI_API_KEY"] = "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"


directory = 'programic'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)



def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


docs = split_docs(documents)
print("Size:", len(docs))

# daha önceden eğitilmiş olan yapay zekayı kullan
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# vectordb'e documentleri yükle
db = Chroma.from_documents(docs, embeddings)

# query = "Express vs Urlencoded?"
# matching_docs = db.similarity_search(query)

# cevap = matching_docs[0]

# print("AI tarafından verilen cevap:", cevap)



from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)


chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

query = input("Soru sor: ")
matching_docs = db.similarity_search(query)
answer =  chain.run(input_documents=matching_docs, question=query)

print("gelen yanıt:", answer)
