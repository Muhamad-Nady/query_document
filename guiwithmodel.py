import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QAction, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog
from PyQt5.QtCore import QTimer,QDateTime, Qt
from PyQt5.QtGui import QTextOption
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,TextSplitter
from langchain.vectorstores import FAISS
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.chains import RetrievalQA
from langchain import  ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Chat GUI')
        self.setGeometry(300, 300, 700, 700)


        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.NoWrap)
        self.text_edit.setWordWrapMode(QTextOption.WrapAnywhere)  # set word wrap mode to WrapAnywhere
        self.text_edit = QTextEdit(self)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # set vertical scroll bar policy

        
        self.text_input = QLineEdit(self)
        self.text_input.returnPressed.connect(self.chat_bot)
        
        self.send_button = QPushButton('Send', self)
        self.send_button.clicked.connect(self.chat_bot)

        self.file_button = QPushButton('Select PDF', self)
        self.file_button.clicked.connect(self.get_pdf)
        llm = self.load_model()
        self.llm = llm
        self.file_name = None
        self.text = None
        self.faiss_index = None
        #self.llm = None        
        
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.text_input)
        hbox.addWidget(self.send_button)
        hbox.addWidget(self.file_button)
        vbox.addWidget(self.text_edit)
        vbox.addLayout(hbox)

        central_widget = QWidget()
        central_widget.setLayout(vbox)
        self.setCentralWidget(central_widget)
        self.show()
    
    
    def chat_bot(self):
        text = self.text_input.text()
        if text:
            self.text_edit.append(f"You: {text}")
            self.text_input.clear()
            if self.faiss_index != None:
                ans = self.query_fun(llm=self.llm, faiss_index=self.faiss_index, query=text)
                result = ans["result"]
                source_doc = ans["source_documents"]
                self.text_edit.append(f"Casey: {result}\n\n")
                for page_content in source_doc:
                    self.text_edit.append(f"{page_content}\n")
            else:
                results = self.assistant(llm=self.llm, query=text)
                self.text_edit.append(f"gpt4all: {results}")
    
    def get_pdf(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","PDF Files (*.pdf)", options=options)
        if file_name:
            directory = os.path.dirname(file_name)
            self.text_edit.append(f"Selected PDF: {file_name}")
            self.text_edit.append(f"Directory: {directory}")
            self.file_name = file_name

            # Implementing the llm functions
            texts = self.pdf_reader(self.file_name)
            faiss_search = self.doc_index(texts=texts)
            self.faiss_index = faiss_search
            

    def pdf_reader(self, filename):
        reader = PdfReader(filename)
        # read data from the file and put them into a variable called raw_text
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 300,
            chunk_overlap  = 100,
            length_function = len,)
        texts = text_splitter.split_text(raw_text)
        return texts

    def doc_index(self, texts):
        # TensorflowHubEmbeddings embedding done
        embeddings = TensorflowHubEmbeddings()
        #select which embeddings we want to use
        faiss_index = FAISS.from_texts(texts, embeddings, chain_type="stuff")
        return faiss_index

    def load_model(self):
        model_path = os.path.join(os.getcwd(), "ggjt-model.bin")
        llm = LlamaCpp(model_path=f"{model_path}", n_ctx=1024, temperature=.05, f16_kv=True, logits_all=False)
        llm.client.verbose = False
        llm.client.temprature = .05
        self.llm = llm
        return llm

    def query_fun(self, llm, faiss_index, query=""):
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,
            just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer in English:"""
        PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={"k":5}),
                                     chain_type_kwargs=chain_type_kwargs,
                                     return_source_documents=True,)
	    #results = qa.run(query)
        ans = qa({"query": query})

        return ans
    

    def assistant(self, llm, query=""):
        template = """Assistant is a large language model trained by OpenAI.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

        {history}
        Human: {human_input}
        Assistant:"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"], 
            template=template
        )


        chatgpt_chain = LLMChain(
            llm=llm, 
            prompt=prompt, 
            memory=ConversationBufferWindowMemory(k=2),
        )

        output = chatgpt_chain.predict(human_input=query)
        return output

if __name__ == '__main__':
    app = QApplication(sys.argv)
    chat_window = ChatWindow()
    sys.exit(app.exec_())
