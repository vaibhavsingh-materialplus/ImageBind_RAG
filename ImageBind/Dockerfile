FROM python:3.9.6
LABEL authors="vaibhav.singh"

WORKDIR /app

# Copy your Python file to the working directory
COPY requirements.txt .
#RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["gradio", "app.py"]

