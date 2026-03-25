# 📝 AI Quiz Generator

A professional web application that generates quiz questions from any document (PDF, PPT, DOCX, TXT) using AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🎯 **Multiple Document Formats**: PDF, PowerPoint, Word, Text
- 🤖 **AI-Powered**: Uses Ollama with llama3 model
- 📊 **Flexible Quiz Types**: Multiple choice or True/False questions
- 🎨 **Modern UI**: Clean, intuitive web interface
- 💾 **Export**: Download generated quizzes
- ⚡ **Fast**: Cached embeddings for quick generation

## 🚀 Quick Start

### 1. Install Ollama

```bash
# Visit https://ollama.ai and install Ollama
# Then pull the llama3 model
ollama pull llama3
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### Web Application (Recommended)

1. **Launch the app**: `streamlit run app.py`
2. **Upload your document** using the sidebar
3. **Configure settings**:
   - Number of questions (1-20)
   - Question type (Multiple Choice or True/False)
4. **Click "Generate Quiz"**
5. **Download** your quiz as a text file

### Command Line Interface

```bash
# Interactive mode
python3 "from langchain.py"

# With arguments
python3 "from langchain.py" document.pdf 5
```

## 📁 Supported File Formats

| Format | Extensions | Status |
|--------|-----------|--------|
| PDF | `.pdf` | ✅ Supported |
| PowerPoint | `.ppt`, `.pptx` | ✅ Supported |
| Word | `.doc`, `.docx` | ✅ Supported |
| Text | `.txt` | ✅ Supported |

## 🎯 Example Output

```
Question 1:
What is supervised learning?

A) Learning from unlabeled data
B) Learning from labeled training data
C) Learning through trial and error
D) Learning without any data

Correct Answer: B
Explanation: Supervised learning uses labeled training data where the algorithm learns from input-output pairs.

---

Question 2:
Which technique is used to evaluate model performance?

A) Overfitting
B) Features
C) Cross-validation
D) Labels

Correct Answer: C
Explanation: Cross-validation is a technique that splits data into multiple folds to assess how well a model generalizes.
```

## 🛠️ Project Structure

```
.
├── app.py                  # Streamlit web application
├── from langchain.py       # CLI version
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
└── sample_lecture.txt     # Sample document
```

## 🔧 Configuration

### Change AI Model

Edit `app.py` or `from langchain.py`:

```python
llm = OllamaLLM(model="llama3")  # Change to any Ollama model
```

### Adjust Number of Retrieved Documents

```python
relevant_docs = vectorstore.similarity_search(query, k=4)  # Change k value
```

## 🐛 Troubleshooting

### "Connection refused" error
- Ensure Ollama is installed and running
- Check if llama3 model is downloaded: `ollama list`
- Download if needed: `ollama pull llama3`

### "File not found" error
- Verify the file path is correct
- Check file permissions

### Slow first run
- First run downloads embedding model (~438MB)
- Subsequent runs will be faster due to caching

### Memory issues
- Reduce number of documents retrieved (lower `k` value)
- Use smaller documents
- Close other applications

## 📦 Dependencies

- **streamlit**: Web application framework
- **langchain**: LLM orchestration
- **ollama**: Local AI model
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Text embeddings
- **unstructured**: Document parsing

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - feel free to use this project for any purpose.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for local AI models
- [LangChain](https://langchain.com) for LLM framework
- [Streamlit](https://streamlit.io) for web framework
