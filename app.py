from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,TextSplitter
from langchain.vectorstores import FAISS
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.chains import RetrievalQA
import sys
import os
from os.path import exists
from os.path import join
def pdf_reader(filename):
    reader = PdfReader('/home/studio-lab-user/langchain/Transmorphosis (3).pdf')
    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 150,
        chunk_overlap  = 50,
        length_function = len,)
    texts = text_splitter.split_text(raw_text)
    return texts
def doc_index(texts):
    ## TensorflowHubEmbeddings embeding dun
    embeddings = TensorflowHubEmbeddings()
    #select which embeddings we want to use
    faiss_index = FAISS.from_texts(texts, embeddings, chain_type="stuff")
    return faiss_index

def download_model():
    if exists(join(os.getcwd(), "ggjt-model.bin")):
        model_path = join(os.getcwd(), "ggjt-model.bin")
        return model_path
    else:
        hf_hub_download(repo_id="LLukas22/gpt4all-lora-quantized-ggjt", 
                        filename="ggjt-model.bin", 
                        local_dir=".")
        model_path = join(os.getcwd(), "ggjt-model.bin")
        return model_path
def load_model(model_path):
    if bool(id(llm)):
        return llm
    else:
        llm = LlamaCpp(model_path=f"{model_path}", n_ctx=1024)
        return llm

if __name__ == '__main__':
    doc_path = sys.argv[1]
    texts = pdf_reader(filename=doc_path)
    fais_search = doc_index(texts=texts)
    model_path = download_model
    llm = load_model(model_path=model_path)
    while(True):
        print("""here is you assistant for document QA 
              please let me know how could i help you,
              + "\n" + for quit the program writr q""", sep="\n") 
        query = input()
        if query == "q":
            break
        else:
            # expose this index in a retriever interface
            retriever = fais_search.as_retriever(search_type="similarity", search_kwargs={"k":2})
            #create a chain to answer questions 
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
            result = qa({"query": query})
            print(query, f"I think it will be{result["result"]}", sep="\n")
        