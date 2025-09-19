import os
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Tuple
from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect
import warnings
from flask import Flask, render_template,request, jsonify
warnings.filterwarnings("ignore")

class MultilingualPDFChatbot:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the multilingual PDF chatbot with embedding model
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        self.pdf_content = ""
        self.user_language = "en"  # Default to English

        # Initialize translator
        self.translator = GoogleTranslator(source='auto', target='en')

        # Language mappings
        self.language_map = {
            "english": "en",
            "hindi": "hi",
            "marathi": "mr",
            "marwari": "mwr"  # Limited support
        }

        self.language_names = {
            "en": "English",
            "hi": "Hindi",
            "mr": "Marathi",
            "mwr": "Marwari"
        }

        # Initialize QA pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad"
        )

        print("Multilingual PDF Chatbot initialized!")
        self.display_language_options()

    def display_language_options(self):
        print("\nSupported Languages:")
        print("1. English")
        print("2. Hindi (हिंदी)")
        print("3. Marathi (मराठी)")
        print("4. Marwari (मारवाड़ी)")
        print("\nUse set_language() method to change language preference.")

    def set_language(self, language: str):
        language = language.lower().strip()
        if language in self.language_map:
            self.user_language = self.language_map[language]
        elif language in self.language_map.values():
            self.user_language = language
        else:
            print(f"Language '{language}' not supported. Available options:")
            for lang_name in self.language_map.keys():
                print(f"- {lang_name}")
            return False

        lang_name = self.language_names[self.user_language]
        print(f"Language set to: {lang_name}")

        if self.user_language == "hi":
            print("भाषा हिंदी में सेट की गई है!")
        elif self.user_language == "mr":
            print("भाषा मराठीत सेट केली आहे!")
        elif self.user_language == "mwr":
            print("भाषा मारवाड़ी में सेट की गई है!")

        return True

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except:
            return "en"

    def translate_text(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """
        Translate text between languages using deep-translator
        """
        try:
            if source_lang == target_lang:
                return text

            # Handle Marwari (limited support)
            if target_lang == "mwr" or source_lang == "mwr":
                if target_lang == "mwr":
                    hindi_text = GoogleTranslator(source=source_lang, target="hi").translate(text)
                    return f"{hindi_text} (मारवाड़ी संदर्भ में)"
                else:
                    return GoogleTranslator(source="hi", target=target_lang).translate(text)

            return GoogleTranslator(source=source_lang, target=target_lang).translate(text)

        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\u0900-\u097F\u0980-\u09FF]', '', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        chunks = []
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        status_msg = "Creating embeddings..."
        if self.user_language != "en":
            status_msg = self.translate_text(status_msg, "en", self.user_language)
        print(status_msg)
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        return np.array(embeddings)

    def find_similar_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if self.embeddings is None:
            return []

        detected_lang = self.detect_language(query)
        query_for_search = query
        if detected_lang != "en":
            query_for_search = self.translate_text(query, detected_lang, "en")

        query_embedding = self.embedding_model.encode([query_for_search])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.chunks[idx], similarities[idx]) for idx in top_indices]

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        context = " ".join(context_chunks)
        detected_lang = self.detect_language(query)
        query_en = query if detected_lang == "en" else self.translate_text(query, detected_lang, "en")

        context_lang = self.detect_language(context)
        context_en = context if context_lang == "en" else self.translate_text(context, context_lang, "en")
        if len(context_en) > 2000:
            context_en = context_en[:2000]

        try:
            result = self.qa_pipeline(question=query_en, context=context_en)
            answer = result['answer']
            if self.user_language != "en":
                answer = self.translate_text(answer, "en", self.user_language)
            return answer
        except Exception as e:
            error_msg = f"I couldn't generate an answer. Error: {e}"
            if self.user_language != "en":
                error_msg = self.translate_text(error_msg, "en", self.user_language)
            return error_msg

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            error_msg = f"Error reading PDF: {e}"
            if self.user_language != "en":
                error_msg = self.translate_text(error_msg, "en", self.user_language)
            print(error_msg)
        return text

    def load_pdf(self, pdf_path: str, chunk_size: int = 500, overlap: int = 50):
        print(f"Loading PDF: {pdf_path}")
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            raise ValueError(self.translate_text("Could not extract text from PDF", "en", self.user_language))

        self.pdf_content = self.clean_text(raw_text)
        print(f"Extracted {len(self.pdf_content)} characters")
        self.chunks = self.chunk_text(self.pdf_content, chunk_size, overlap)
        print(f"Created {len(self.chunks)} chunks")
        self.embeddings = self.create_embeddings(self.chunks)
        print("PDF processing complete!")

    def ask_question(self, question: str, top_k: int = 3) -> dict:
        if not self.chunks:
            return {"error": self.translate_text("No PDF loaded. Please load a PDF first.", "en", self.user_language)}

        similar_chunks = self.find_similar_chunks(question, top_k)
        if not similar_chunks:
            return {"error": self.translate_text("Could not find relevant information.", "en", self.user_language)}

        context_chunks = [chunk[0] for chunk in similar_chunks]
        answer = self.generate_answer(question, context_chunks)

        return {
            "answer": answer,
            "similar_chunks": similar_chunks,
            "confidence": similar_chunks[0][1] if similar_chunks else 0,
            "question_language": self.detect_language(question),
            "answer_language": self.user_language
        }

    def interactive_language_setup(self):
        print("\n" + "="*60)
        print("🌍 MULTILINGUAL PDF CHATBOT 🌍")
        print("="*60)
        print("\nPlease select your preferred language:")
        print("1. English\n2. हिंदी (Hindi)\n3. मराठी (Marathi)\n4. मारवाड़ी (Marwari)")

        while True:
            choice = input("Enter your choice (1-4): ").strip()
            if choice == "1":
                self.set_language("english")
                break
            elif choice == "2":
                self.set_language("hindi")
                break
            elif choice == "3":
                self.set_language("marathi")
                break
            elif choice == "4":
                self.set_language("marwari")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")


def main():
    chatbot = MultilingualPDFChatbot()
    chatbot.interactive_language_setup()
    pdf_path = input("Enter PDF file path: ").strip()
    try:
        chatbot.load_pdf(pdf_path)

        ready_msg = "PDF Chatbot Ready! Ask questions about your document."
        quit_msg = "Type 'quit', 'exit', or 'q' to exit."
        lang_msg = "Type 'lang' to change language."

        if chatbot.user_language != "en":
            ready_msg = chatbot.translate_text(ready_msg, "en", chatbot.user_language)
            quit_msg = chatbot.translate_text(quit_msg, "en", chatbot.user_language)
            lang_msg = chatbot.translate_text(lang_msg, "en", chatbot.user_language)

        print("\n" + "="*60)
        print(ready_msg)
        print(quit_msg)
        print(lang_msg)
        print("="*60 + "\n")

        while True:
            if chatbot.user_language == "hi":
                question = input("आपका प्रश्न: ").strip()
            elif chatbot.user_language == "mr":
                question = input("तुमचा प्रश्न: ").strip()
            elif chatbot.user_language == "mwr":
                question = input("थारो सवाल: ").strip()
            else:
                question = input("Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q', 'बाहर', 'बंद']:
                break
            if question.lower() in ['lang', 'language', 'भाषा']:
                chatbot.interactive_language_setup()
                continue
            if not question:
                continue

            result = chatbot.ask_question(question)
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"\n📝 Answer: {result['answer']}")
                print(f"🎯 Confidence: {result['confidence']:.3f}")
                print("-" * 60)

    except Exception as e:
        print(f"Error: {e}")

app = Flask(__name__)
chatbot = MultilingualPDFChatbot()
chatbot.set_language("english")  # Default language; can be modified to user preference
pdf_path = "data/rizvi.pdf"  
chatbot.load_pdf(pdf_path)
@app.route('/')
def home():
    return render_template('index.html', name="Shikha")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    print(data)
    user_message = data.get('chat', '')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400
  # Replace with actual PDF path or user-uploaded file
    try:
        result = chatbot.ask_question(user_message)
        print(result)
        if "error" in result:
            return jsonify({"error": result['error']}), 500

        safe_result = {
            "answer": result.get("answer"),
            "confidence": float(result.get("confidence", 0)),   # convert np.float32 -> float
        }

        return jsonify(safe_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)

# Example usage for different languages
"""
# Example usage:
chatbot = MultilingualPDFChatbot()

# Set language to Hindi
chatbot.set_language("hindi")

# Load PDF
chatbot.load_pdf("document.pdf")

# Ask question in Hindi
result = chatbot.ask_question("यह दस्तावेज़ किस बारे में है?")
print(result["answer"])

# Ask question in English (will be answered in Hindi)
result = chatbot.ask_question("What is the main topic?")
print(result["answer"])

# Change to Marathi
chatbot.set_language("marathi")
result = chatbot.ask_question("हा दस्तऐवज कशाबद्दल आहे?")
print(result["answer"])
"""