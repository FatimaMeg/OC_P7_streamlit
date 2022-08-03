# backend/Dockerfile

# Set base image (Python 3.9)
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy list of backend requirements
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Copy files to working directory (into a newly created /app directory)
COPY . /app

# Expose port
EXPOSE 8000

# Execute command when container starts
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]