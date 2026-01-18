import os, re, time, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI


# Preprocessing for Documents
class DocumentationProcessor:
    def __init__(self , Document_Path : str):
        self.Document_Path = Document_Path

    def clean_text(self , text : str):
        text = re.sub(r"[🌟🎯📌💡🩺📞💳❓📜🔒⏰📝*#\-]+", " ", text)
        text = re.sub(r"[\"“”]+", "\"", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def load_and_split_chunks(self , chunk_size = 500 , chunk_overlap = 50):
        loader = TextLoader(self.Document_Path , encoding = "utf-8")
        documents = loader.load()

        cleaned_texts = []
        for doc in documents :
            cleaned_text = self.clean_text(doc.page_content)
# remove empty lines and chunks that very small
            if cleaned_text and len(cleaned_text) > 10 :
                cleaned_texts.append(cleaned_text)

        all_text = " ".join(cleaned_texts)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", "،", ".", " "]
        )
        chunks = text_splitter.create_documents([all_text])

        cleaned_chunks = []
        for chunk in chunks : 
            cleaned_content = self.clean_text(chunk.page_content)
            if cleaned_content and len(cleaned_content) > 20 : 
                cleaned_chunks.append(cleaned_content)


        return text_splitter.create_documents(cleaned_chunks)
    
# Class for embeding and Faiss
class VectorStore: 
    def __init__(self , model_name = "OmarAlsaabi/e5-base-mlqa-finetuned-arabic-for-rag"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.docs = None
        self.embeddings = None

    def build_index(self , docs):
        self.docs = docs
        self.embeddings = np.array([self.embedder.encode(doc.page_content) for doc in docs]).astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def save_index(self , index_path = "index.faiss" , docs_path="docs.pkl", emb_path="embeddings.npy"):
        #save embedding and documents and faiss
        faiss.write_index(self.index , index_path)
        with open(docs_path , "wb") as f :
            pickle.dump(self.docs , f )
        np.save(emb_path , self.embeddings)

    def load_index(self, index_path="index.faiss", docs_path="docs.pkl", emb_path="embeddings.npy"):
        if all(os.path.exists(path)for path in [index_path , docs_path , emb_path]):
            self.index = faiss.read_index(index_path)
            with open(docs_path ,"rb")as f : 
                self.docs = pickle.load(f)
            self.embeddings = np.load(emb_path)
            return True
        return False
    
    def retrieve(self , query ,k=3):
        ## return chunks that more related for question 
        query_embedding = np.array(self.embedder.encode([query])).astype("float32")
        D,I = self.index.search(query_embedding , k )

        relevant_text = []
        for i in I[0]:
            if i < len(self.docs):
                clean_text = self.clean_context(self.docs[i].page_content)
                relevant_text.append(clean_text)
            # طباعة النتائج المسترجعة للتشخيص
        print(f"🔍 Query: {query}")
        print(f"📄 Retrieved {len(relevant_text)} chunks:")
        for idx, text in enumerate(relevant_text):
            print(f"  {idx+1}. {text[:100]}...")
        
        return " ".join(relevant_text)
    

    def clean_context(self, text):
        text = re.sub(r"\*+", " ", text) 
        text = re.sub(r"\s+", " ", text).strip()
        return text
    


class ArabicRAGGPT_class:
    def __init__(self , document_path , api_key ,index_path="index.faiss",
                  docs_path="docs.pkl", emb_path="embeddings.npy"):
        self.document_path = document_path
        self.api_key = api_key
        self.client =  OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.vestore = VectorStore()
        self.initialized = False
        self.processor = DocumentationProcessor(document_path)

        self.index_path = index_path
        self.docs_path = docs_path
        self.emb_path = emb_path


    def initialize(self) : 
        if self.initialized : 
            return
        starttime = time.time()
        loaded_files = self.vestore.load_index(self.index_path , self.docs_path , self.emb_path )

        if not loaded_files :
            print("Hiiiiiii")
            if not os.path.exists(self.document_path):
                raise FileNotFoundError(f"المستند الأصلي غير موجود: {self.document_path}")
            
            docs = self.processor.load_and_split_chunks()
            self.vestore.build_index(docs)
            self.vestore.save_index(self.index_path , self.docs_path ,  self.emb_path   )
        else : 
            self.initialized = True
            print(f"Time : {time.time() - starttime:.2f}")

    def generate_answer(self, query, context):
            """توليد إجابة من GPT - نظيفة وواضحة"""
            prompt = f"""
    المعلومات المرجعية:
    {context}

    السؤال: {query}

    المطلوب: قدم إجابة واضحة ومباشرة باللغة العربية بناءً على المعلومات المذكورة أعلاه فقط.
    """
            response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
            {"role": "system", "content": "أنت مساعد عربي محترف. قدم إجابات دقيقة وواضحة بناءً على المعلومات المعطاة."},
            {"role": "user", "content": prompt}
            ],
                    temperature=0.3,  # أقل لنتائج أكثر استقراراً
                    max_tokens=500
                )
            return self.clean_answer(response.choices[0].message.content.strip())

    def clean_answer(self , answer): 
        answer = re.sub(r"\*+", "", answer)
        answer = re.sub(r"#+", "", answer)
        answer = re.sub(r"-+", " - ", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer
    
    def ask_question(self , query):
        if not self.initialized : 
            self.initialize()

        print(f"Question : {query}")
        starttime = time.time()
        context = self.vestore.retrieve(query , k=3)
        answer = self.generate_answer(query , context)
        process_time = time.time() - starttime
        print(f"time : {process_time}")
        return answer