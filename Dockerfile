# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Command to run your app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

