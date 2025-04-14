#!/bin/bash

# Create necessary directories
mkdir -p ~/.streamlit/

# Create config.toml file for Streamlit
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Install requirements
pip install -r requirements.txt 