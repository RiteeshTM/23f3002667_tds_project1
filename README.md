# FastAPI Automation Agent

## Overview
This project is a FastAPI-based automation agent designed to execute predefined tasks based on natural language instructions. It supports various operations such as formatting files, querying databases, extracting text from images, running scripts, and more.

## Features
- Exposes API endpoints for executing automated tasks.
- Uses gpt-4o-mini for task classification.
- Supports file processing, text extraction, and database queries.
- Runs within a Docker container for portability.
- Implements CORS middleware for cross-origin requests.

## Requirements
- Python >= 3.13
- The following dependencies:
  ```plaintext
  fastapi
  uvicorn
  requests
  pandas
  openai
  Pillow
  pytesseract
  sentence-transformers
  ```

## Installation
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```sh
   export AIPROXY_TOKEN=<your_openai_api_key>
   ```

## Usage
### Start the API server
Run the FastAPI server using Uvicorn:
```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints
#### Home
```http
GET /
```
Response:
```json
{"message": "Yay TDS Tuesday is awesome."}
```

#### Read a File
```http
GET /read?path=<file_path>
```
Retrieves the content of a specified file.

#### Run a Task
```http
POST /run?task=<task_description>
```
Executes a specified task based on classification.

### Example Tasks
- Install `uv` and run `datagen.py`
- Format files using Prettier
- Count the number of Wednesdays in a dataset
- Sort contacts
- Extract email senders, credit card details, and H1 titles
- Compute total sales for gold tickets

## Running in Docker
1. Build the Docker image:
   ```sh
   docker build -t fastapi-agent .
   ```
   or use podman 
   ```sh
   podman build -t fastapi-agent .
   ```
2. Run the container:
   ```sh
   docker run -p 8000:8000 --env AIPROXY_TOKEN=<your_openai_api_key> fastapi-agent
   ```
    or use podman

    ```sh
   podman run -p 8000:8000 --env AIPROXY_TOKEN=<your_openai_api_key> fastapi-agent
   ```

## Contributing
Pull requests are welcome. Please follow best practices and submit an issue before making significant changes.

## License
This project is licensed under the MIT License.

