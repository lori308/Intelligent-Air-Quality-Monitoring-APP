To run the application correctly, follow these simple instructions:
1. Install dependencies:

# Activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate # Windows

# or
source venv/bin/activate # macOS/Linux

# Install the necessary libraries

pip install streamlit pandas plotly requests folium openai psutil pymupdf llama-index-core llama-index-vector-stores-chroma llama-index-embeddings-cohere chromadb scipy numpy cohere python-dotenv

2. Start the application, via the terminal of your IDE

streamlit run aia_gj.py

The interface will be available in your browser at: http://localhost:8501

3. Enter developer mode in the pop-up window on the left

To access advanced tools from the sidebar:

• Username: admin
• Password: admin123

Best regards,
Lorand Gjoshi
