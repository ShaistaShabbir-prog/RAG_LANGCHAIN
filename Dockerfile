# Step 1: Use an official Python image from the Docker Hub
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements.txt file (contains your dependencies)
COPY requirements.txt ./

# Step 4: Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the entire FastAPI project into the container
COPY . .

# Step 6: Expose the port the FastAPI application runs on (by default, FastAPI runs on port 8000)
EXPOSE 10500

# Step 7: Define the command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10500"]

