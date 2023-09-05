## building a gui for extractive QA task + chat assistant
# importing the libraries 
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QAction, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, QLabel
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
from langchain.llms import GPT4All

#create the gui window
class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Window title
        self.setWindowTitle('Chat GUI')
        # Window dimensions
        self.setGeometry(300, 100, 800, 800)

        # Text box for input text
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.NoWrap)
        self.text_edit.setWordWrapMode(QTextOption.WrapAnywhere)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # Send the text in text box with returnPressed and run the chatbot function
        self.text_input = QLineEdit(self)
        self.text_input.returnPressed.connect(self.chat_bot)

        # Send the text in text box with send button
        self.send_button = QPushButton('Send', self)
        self.send_button.clicked.connect(self.chat_bot)

        # Button to select the PDFs for query and run function
        self.file_button = QPushButton('Select PDF', self)
        self.file_button.clicked.connect(self.get_pdf)


        # Add a new QLineEdit widget to the GUI for entering the temperature value
        self.temperature_input = QLineEdit(self)
        self.temperature_label = QLabel("Temperature:", self)
        self.temperature_input.returnPressed.connect(self.set_temperature)
        # adding set_temprsture button
        self.set_temp = QPushButton('set_temp', self)
        self.set_temp.clicked.connect(self.set_temperature)


        vbox = QVBoxLayout()

        hbox_temp = QHBoxLayout()
        hbox_temp.addWidget(self.temperature_label)
        hbox_temp.addWidget(self.temperature_input)
        hbox_temp.addWidget(self.set_temp)

        hbox_input = QHBoxLayout()
        hbox_input.addWidget(self.text_input)
        hbox_input.addWidget(self.send_button)
        hbox_input.addWidget(self.file_button)

        vbox.addWidget(self.text_edit)
        vbox.addLayout(hbox_temp)
        vbox.addLayout(hbox_input)

        # Modify the layout to include the temperature label and input
        central_widget = QWidget()
        central_widget.setLayout(vbox)
        self.setCentralWidget(central_widget)
        self.show()
        #asking for set temp
        self.text_edit.append("please set you temp before starting otherwise it will be 0.7 as default until you set it to new value")
	
        #set llm values to null
        self.llm = None	
        # list of pdf file name
        self.file_names = None
        # list of  pdfs chrcuncked text
        self.text = None
        #declear the fias index for semilartiy search
        self.faiss_index = None    
    
    # chat function
    def chat_bot(self):
    	# Check llm value:
        if not self.llm:
            self.llm = self.load_model()

        # input text of input box
        text = self.text_input.text()
        if text:
            # print the input text in chat window
            self.text_edit.append(f"You: {text}")
            # clear the text box
            self.text_input.clear()
            # create a fais_index after selecting pdf
            if self.faiss_index != None:
                # run the query function to extract answer out of text
                ans = self.query_fun(llm=self.llm, faiss_index=self.faiss_index, query=text)
                # return the query result
                result = ans["result"]
                # return the source document
                source_doc = ans["source_documents"]
                # senf the result and source text to chat widget
                self.text_edit.append(f"Casey: {result}\n\n")
                for page_content in source_doc:
                    self.text_edit.append(f"{page_content}\n")
            # if you don't select a pdf run the assistant function
            else:
                results = self.assistant(llm=self.llm, query=text)
                self.text_edit.append(f"gpt4all: {results}")
    # set temprature function
    def set_temperature(self):
        temperature = float(self.temperature_input.text())
        llm = self.load_model(temprature=temperature)
        self.llm = llm
        return llm

    # that function is to select the pdfs
    def get_pdf(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select PDF Files", "", "PDF Files (*.pdf)", options=options)
        if file_names:
            directory = os.path.dirname(file_names[0])
            for file_name in file_names:
                self.text_edit.append(file_name)
                self.text_edit.append(f"Directory: {directory}")
            self.file_names = file_names
            
            # run all function for convert pdfs to text adding fais_indexing
            # Implementing the llm functions
            texts = self.pdf_reader(self.file_names)
            faiss_search = self.doc_index(texts=texts)
            self.faiss_index = faiss_search
            
    # Read the pdfs file as text
    def pdf_reader(self, filenames):
        # read data from the file and put them into a variable called raw_text
        raw_text = ''
        # first read the pdf file by file
        for filename in filenames:
            reader = PdfReader(filename)
            # read data from the file and put them into a variable called raw_text
            # read each page in every file and save it to text variable
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += f" {text}"
        #split the text to chunks
        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 800,
            chunk_overlap  = 300,
            length_function = len,)
        texts = text_splitter.split_text(raw_text)
        return texts
    
    # embedding text using tensorflow texthub with defaut model universal-sentence-encoder-multilingual dim (n, 512)
    def doc_index(self, texts):
        # TensorflowHubEmbeddings embedding done
        model_path = os.path.join(os.getcwd(), "embed_model")
        embeddings = TensorflowHubEmbeddings(model_url=f"{model_path}")
        #select which embeddings we want to use
        faiss_index = FAISS.from_texts(texts, embeddings, chain_type="stuff")
        return faiss_index
    # load the model using llama-ccp-python binding 
    def load_model(self, temprature=0.7):
        self.text_edit.append(f"you have set temprature to {temprature}")
        self.text_edit.append(f"Running ggml-gpt4all-j-v1.3-groovy.bin")
    	# 4-bit quantized_model with ggml lora-model
    	# hugging face repo url of model https://huggingface.co/LLukas22/gpt4all-lora-quantized-ggjt/tree/main
        # get model path
        model_path = "./ggml-gpt4all-j-v1.3-groovy.bin"
        #load the model
        llm = GPT4All(model=model_path, n_ctx=1000, backend="gptj", verbose=False)        # model temp
        ##llm.client.temprature = temprature
        self.llm = llm
        return llm

    # function of Extractive QA, using that prompt to extarct the query out of source text
    def query_fun(self, llm, faiss_index, query=""):
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,
                        just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer should be based on the information provided in the context:"""
        PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={"k":4}),
                                     chain_type_kwargs=chain_type_kwargs,
                                     return_source_documents=True,)
	    #results = qa.run(query)
        ans = qa({"query": query})

        return ans
    

    # function of assistant chat, using that prompt to extarct the query out of source text
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
