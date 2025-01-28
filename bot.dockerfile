# Base image
FROM python:3.11-slim

# Install Python
#RUN apt update && \
#    apt install --no-install-recommends -y build-essential gcc && \
#    apt clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt requirements.txt
COPY app2.py app2.py
COPY pyproject.toml pyproject.toml
#COPY srtm/ strm/
COPY data/ data/

# set working directory
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir 

# mounts local pip cache to avoid redownloading packages
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt  
#RUN pip install . --no-deps --no-cache-dir


# Expose the port your Streamlit app listens on
#EXPOSE 8000, 8501

# Command to run the Streamlit application
#CMD ["python", "-m", "streamlit", "run", "app2.py", "--server.port=8000", "--server.address=0.0.0.0"]
#CMD ["streamlit", "run", "app2.py"]
ENTRYPOINT ["python", "-m", "streamlit", "run", "app2.py", "--server.port=8000", "--server.address=0.0.0.0"]