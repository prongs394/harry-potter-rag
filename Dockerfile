# 1. Minimal Linux base with Python 3.10
FROM python:3.10-slim

# 2. Define the container working directory
WORKDIR /app

# 3. Copy dependencies list first to utilize Docker layer cache
COPY project_requirements.txt .

# 4. Install required libraries without caching wheels
RUN pip install --no-cache-dir -r project_requirements.txt

# 5. Copy the FastAPI server script
COPY server.py .

# 6. Expose the port used by the server
EXPOSE 8000

# 7. Default command to start the FastAPI service
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]