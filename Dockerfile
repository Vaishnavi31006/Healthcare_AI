FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

# Use app.py (Flask server) as the entry point — stays alive on port 7860
# as required by HuggingFace Spaces Docker SDK.
CMD ["python", "-u", "app.py"]
