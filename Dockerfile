# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --default-timeout=200 -r requirements.txt

# 4. Copy the source code into the container
COPY src/ ./src/

# 5. Copy BOTH the MLflow tracking folder and the SQLite database
COPY mlruns/ ./mlruns/
COPY mlflow.db .

# 6. NEW: Tell the API inside the container to look at the database
ENV MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Define the command to run the FastAPI server
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]