FROM python:3.11-slim

WORKDIR /app

# Copy code and requirements
COPY Requirements.txt ./
COPY process_pdfs.py ./
# Also copy your schema file, and any utils you use
COPY sample_datasets/schema/output_schema.json ./schema/
COPY sample_datasets/schema/create_output_schema.py ./

# Install requirements
RUN pip install --no-cache-dir -r Requirements.txt

# (Optional) Set environment variables
ENV PYTHONUNBUFFERED=1

CMD ["python", "process_pdfs.py"]
