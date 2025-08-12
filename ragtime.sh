# 0) (optional) make a clean venv
python3 -m venv .ragtime && source .ragtime/bin/activate

# 1) get the repo
git clone https://github.com/trobjohn/ragtime && cd ragtime

# 2) install deps
pip install -r requirements.txt
# if torch moans on CPU-only:
# pip install --extra-index-url https://download.pytorch.org/whl/cpu torch

# 3) install Ollama + tiny model
# (mac: brew install ollama; linux: curl -fsSL https://ollama.com/install.sh | sh)
ollama pull gemma:2b
ollama run gemma:2b   # keep this terminal open, or run it in background

# 4) add a SMALL test corpus (start tiny!)
mkdir -p data/notes data/extras
# copy 1–2 .ipynb to data/notes/ and 1 PDF (e.g., ESL) to data/extras/

# 5) build the index
python ingest.py   # (or ingest_echo.py)

# 6) launch the app
streamlit run serve.py
