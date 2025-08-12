

## Activate your virtual environment for this situation
source .venv/bin/activate 

## Install required Python packages
pip3 install -r requirements.txt

## Install and run your local LLM

# curl -fsSL https://ollama.com/install.sh | sh # Install ollama
#ollama pull gemma:2b   # smallest viable model 

ollama run gemma:2b

## Pull your content
python ingest.py

## Connect to your LLM
streamlit run serve.py