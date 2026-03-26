# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install setuptools first (mlflow still imports pkg_resources)
RUN pip install --no-cache-dir "setuptools<81" \
    && python -c "import pkg_resources"

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files ke container
COPY . .

# Expose port untuk Render
EXPOSE 8000

# Run Flask dengan port dinamis dari Railway/Render
CMD gunicorn -b 0.0.0.0:${PORT:-8000} app:app
