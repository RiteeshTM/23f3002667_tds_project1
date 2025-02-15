#/// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "requests",
#     "pandas",
#     "openai",
#     "Pillow",
#     "pytesseract",
#     "sentence-transformers",
# ]
#///

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import subprocess
import os
import json
import re
from datetime import datetime
import openai
# from PIL import Image
# import pytesseract
# from sentence_transformers import SentenceTransformer, util
# import pandas as pd


app = FastAPI()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Yay TDS Tuesday is awesome."}

@app.get("/read")
def read_file(path: str):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        raise HTTPException(status_code=404, detail="File doesn't exist")
# Define task categories explicitly
TASK_CATEGORIES = {
    "install uv and run datagen": "execute_task_A1",
    "format with prettier": "execute_task_A2",
    "count wednesdays": "execute_task_A3",
    "sort contacts": "execute_task_A4",
    "recent log files": "execute_task_A5",
    "extract h1 titles": "execute_task_A6",
    "extract sender email": "execute_task_A7",
    "extract credit card": "execute_task_A8",
    "find similar comments": "execute_task_A9",
    "total sales gold tickets": "execute_task_A10"
}


api_key = os.getenv("AIPROXY_TOKEN")  # Fetch token from environment
if not api_key:
    raise ValueError("AIPROXY_TOKEN is not set!")

client = openai.OpenAI(api_key=api_key)

def classify_task(task: str) -> str:
    """Use GPT-4o to classify the task into predefined categories."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a task classifier. Based on the given user input, return exactly one of these categories:\n"
                        + "\n".join(f"- {key}" for key in TASK_CATEGORIES.keys())
                        + "\nIf the task does not match any category, return 'unknown'."
                    ),
                },
                {"role": "user", "content": task},
            ],
        )
        
        # Extracting response content correctly
        category = response.choices[0].message.content.strip().lower()
        
        return TASK_CATEGORIES.get(category, "unknown")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to classify task: {str(e)}")

@app.post("/run")
def task_runner(task: str):#, user_email: str = Query(None, description="User's email")):
    try:
        task_lower =  classify_task(task).lower()

        if "install uv" in task_lower and "run datagen.py" in task_lower:
            return execute_task_A1("23f3002667@ds.study.iitm.ac.in")

        elif "format" in task_lower and "prettier" in task_lower:
            return execute_task_A2()

        elif "count wednesdays" in task_lower or "how many" in task_lower:
            return execute_task_A3()

        elif "sort contacts" in task_lower:
            return execute_task_A4()

        elif "recent log files" in task_lower:
            return execute_task_A5()

        elif "extract h1 titles" in task_lower:
            return execute_task_A6()

        elif "extract sender email" in task_lower:
            return execute_task_A7()

        elif "extract credit card" in task_lower:
            return execute_task_A8()

        elif "find similar comments" in task_lower:
            return execute_task_A9()

        elif "total sales gold tickets" in task_lower:
            return execute_task_A10()

        else:
            raise HTTPException(status_code=400, detail="Task not recognized")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def execute_task_A1(user_email):
    """ Task A1: Install 'uv' and run datagen.py with user_email """
    if not user_email:
        raise HTTPException(status_code=400, detail="User email is required")

    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        subprocess.run(["pip", "install", "uv"], check=True)

    script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    script_path = "/tmp/datagen.py"

    response = requests.get(script_url)
    if response.status_code == 200:
        with open(script_path, "w") as f:
            f.write(response.text)
    else:
        raise HTTPException(status_code=500, detail="Failed to download datagen.py")

    subprocess.run(["python", script_path, user_email], check=True)
    return {"status": "success", "task": "A1 executed"}


def execute_task_A2():
    """ Task A2: Format /data/format.md using Prettier """
    file_path = "/data/format.md"
    prettier_version = "3.4.2"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="format.md not found")

    subprocess.run(["npx", f"prettier@{prettier_version}", "--write", file_path], check=True)
    return {"status": "success", "task": "A2 executed"}


def execute_task_A3():
    """ Task A3: Count number of Wednesdays in /data/dates.txt """
    input_path = "/data/dates.txt"
    output_path = "/data/dates-wednesdays.txt"

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="dates.txt not found")

    with open(input_path, "r") as f:
        dates = f.readlines()

    count = sum(1 for date in dates if datetime.strptime(date.strip(), "%Y-%m-%d").weekday() == 2)

    with open(output_path, "w") as f:
        f.write(str(count))

    return {"status": "success", "task": "A3 executed", "wednesdays": count}


def execute_task_A4():
    """ Task A4: Sort contacts.json by last_name, then first_name """
    input_path = "/data/contacts.json"
    output_path = "/data/contacts-sorted.json"

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="contacts.json not found")

    with open(input_path, "r") as f:
        contacts = json.load(f)

    contacts.sort(key=lambda x: (x["last_name"], x["first_name"]))

    with open(output_path, "w") as f:
        json.dump(contacts, f, indent=2)

    return {"status": "success", "task": "A4 executed"}


def execute_task_A5():
    """ Task A5: Get first line of 10 most recent .log files """
    log_dir = "/data/logs/"
    output_path = "/data/logs-recent.txt"

    logs = sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".log")],
        key=lambda x: os.path.getmtime(os.path.join(log_dir, x)),
        reverse=True
    )[:10]

    with open(output_path, "w") as f:
        for log in logs:
            with open(os.path.join(log_dir, log), "r") as log_file:
                first_line = log_file.readline().strip()
                f.write(first_line + "\n")

    return {"status": "success", "task": "A5 executed"}


def execute_task_A6():
    """ Task A6: Extract first occurrence of H1 titles in .md files """
    docs_dir = "/data/docs/"
    output_path = "/data/docs/index.json"
    index = {}

    for file in os.listdir(docs_dir):
        if file.endswith(".md"):
            with open(os.path.join(docs_dir, file), "r") as f:
                content = f.readlines()
            for line in content:
                if line.startswith("# "):
                    index[file] = line.strip("# ").strip()
                    break

    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

    return {"status": "success", "task": "A6 executed"}

def execute_task_A7():
    """ Task A7: Extract sender email from email.txt """
    input_path = "/data/email.txt"
    output_path = "/data/email-sender.txt"

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="email.txt not found")

    with open(input_path, "r") as f:
        content = f.read()

    email_match = re.search(r"From: (.+?@\S+)", content)
    if not email_match:
        raise HTTPException(status_code=500, detail="No email found in email.txt")

    sender_email = email_match.group(1)

    with open(output_path, "w") as f:
        f.write(sender_email)

    return {"status": "success", "task": "A7 executed", "sender_email": sender_email}


def execute_task_A8():
    """ Task A8: Extract credit card number from credit-card.png """
    input_path = "/data/credit-card.png"
    output_path = "/data/credit-card.txt"

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="credit-card.png not found")

    img = Image.open(input_path)
    extracted_text = pytesseract.image_to_string(img)

    cc_match = re.search(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", extracted_text)
    if not cc_match:
        raise HTTPException(status_code=500, detail="No credit card number found")

    cc_number = cc_match.group(0)

    with open(output_path, "w") as f:
        f.write(cc_number)

    return {"status": "success", "task": "A8 executed", "credit_card": cc_number}


def execute_task_A9():
    """ Task A9: Find most similar pair of comments using sentence embeddings """
    input_path = "/data/comments.json"
    output_path = "/data/comments-similar.json"

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="comments.json not found")

    with open(input_path, "r") as f:
        comments = json.load(f)

    if len(comments) < 2:
        raise HTTPException(status_code=400, detail="Not enough comments to compare")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(comments, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    max_sim = -1
    best_pair = None

    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            sim_score = similarity_matrix[i][j].item()
            if sim_score > max_sim:
                max_sim = sim_score
                best_pair = (comments[i], comments[j])

    if not best_pair:
        raise HTTPException(status_code=500, detail="No similar comments found")

    with open(output_path, "w") as f:
        json.dump({"comment_1": best_pair[0], "comment_2": best_pair[1], "similarity": max_sim}, f, indent=2)

    return {"status": "success", "task": "A9 executed", "similar_comments": best_pair, "similarity": max_sim}


def execute_task_A10():
    """ Task A10: Compute total sales of 'Gold' ticket type """
    input_path = "/data/tickets.csv"
    output_path = "/data/tickets-gold.txt"

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="tickets.csv not found")

    df = pd.read_csv(input_path)

    if "ticket_type" not in df.columns or "amount" not in df.columns:
        raise HTTPException(status_code=500, detail="CSV format incorrect")

    gold_sales = df[df["ticket_type"] == "Gold"]["amount"].sum()

    with open(output_path, "w") as f:
        f.write(str(gold_sales))

    return {"status": "success", "task": "A10 executed", "gold_sales": gold_sales}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
