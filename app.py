import os
import tempfile
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, CallbackContext

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader  

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GROQ_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("API keys not set in the environment variables.")

# Initialize LLM and Embeddings
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

async def start(update: Update, context: CallbackContext):
    """Send a welcome message with options."""
    keyboard = [
        [InlineKeyboardButton("📂 Upload Resume", callback_data="upload_resume")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Welcome to the Resume Interview Bot! 🤖\n"
        "Upload your resume, and I'll generate interview questions for you.",
        reply_markup=reply_markup
    )

async def button_click(update: Update, context: CallbackContext):
    """Handle button clicks."""
    query = update.callback_query
    await query.answer()

    if query.data == "upload_resume":
        await query.message.reply_text("Please upload your resume (PDF format) 📄")
    elif query.data == "help":
        await query.message.reply_text("ℹ️ This bot helps generate interview questions based on your resume. Upload a PDF to get started!")

async def handle_document(update: Update, context: CallbackContext):
    """Handle PDF resume upload and process it."""
    document = update.message.document

    if document.mime_type != "application/pdf":
        await update.message.reply_text("⚠️ Please upload a valid PDF resume.")
        return

    file_id = document.file_id
    file = await context.bot.get_file(file_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        await file.download_to_drive(temp_file.name)
        temp_file_path = temp_file.name

    await update.message.reply_text("Processing your resume... ⏳")

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)

        vector_store = FAISS.from_documents(split_documents, embeddings)
        retriever = vector_store.as_retriever()

        # Generate Questions
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
        )

        prompt = f"""
        Based on the following resume content, generate a comprehensive set of interview questions.
        The questions should cover various aspects such as skills, communication abilities, technical knowledge,
        and tricky or unconventional questions. Here is the resume content:
        
        {split_documents}

        Please ensure that the questions are relevant and varied.
        """

        response = qa_chain.run(prompt)

        await update.message.reply_text("✅ Here are your interview questions:\n\n")
        
        # Split long messages (Telegram has a 4096-character limit)
        for i in range(0, len(response), 4000):
            await update.message.reply_text(response[i:i+4000])

    except Exception as e:
        await update.message.reply_text(f"❌ Error processing resume: {e}")

def main():
    """Start the Telegram bot."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_click))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))

    application.run_polling()

if __name__ == "__main__":
    main()
