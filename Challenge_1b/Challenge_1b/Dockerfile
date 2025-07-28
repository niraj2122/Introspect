# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# STEP 1: Install the CPU-only version of PyTorch directly.
# This is the most important step. We are forcing the install from the correct URL.
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# STEP 2: Copy the requirements file and install the REST of the packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Pre-download the model to bake it into the image ---
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application files
COPY . .

# Define the command to run the application
CMD ["python", "main.py", "--collection", "/app/Collection_1"]
