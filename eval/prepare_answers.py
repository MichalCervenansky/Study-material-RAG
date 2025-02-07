import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import json
import csv
import time
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"
SAMPLE_PDF_PATH = "sample.pdf"
QUESTIONS_CSV_PATH = "questions.csv"
RESULTS_JSON_PATH = "results/results.json"



def clear_database():
    """Clear all documents from the database"""
    response = requests.post(f"{BACKEND_URL}/documents/clear")
    if response.status_code != 200:
        raise Exception("Failed to clear database")
    print("Database cleared successfully")

def upload_pdf():
    """Upload the sample PDF to the database"""
    if not os.path.exists(SAMPLE_PDF_PATH):
        raise FileNotFoundError(f"PDF file not found at {SAMPLE_PDF_PATH}")
    
    try:
        # Upload PDF file directly
        with open(SAMPLE_PDF_PATH, 'rb') as pdf_file:
            files = [('files', pdf_file)]
            response = requests.post(f"{BACKEND_URL}/documents/upload", files=files)
            
        if response.status_code != 200:
            raise Exception(f"Failed to upload PDF. Status code: {response.status_code}. Response: {response.text}")
        
        response_data = response.json()
        if response_data.get('status') != 'success':
            raise Exception(f"Failed to upload PDF: {response_data.get('message', 'Unknown error')}")
            
        print("PDF uploaded successfully")
    except Exception as e:
        print(f"Error during PDF upload: {str(e)}")
        raise

def read_questions():
    """Read questions from CSV file"""
    questions = []
    with open(QUESTIONS_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "id": row["id"],
                "question": row["question"]
            })
    return questions

def query_backend(question):
    """Query the backend with a question and get the response"""
    query_data = {
        "question": question,
        "messages": [],
        "previous_chunks": []
    }
    
    response = requests.post(
        f"{BACKEND_URL}/query",
        json=query_data,
        headers={"Accept": "text/event-stream"},
        stream=True
    )
    
    full_answer = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "answer" in data:
                        full_answer += data["answer"]
                except json.JSONDecodeError:
                    continue
    
    return full_answer.strip()

def test_backend_connection():
    """Test if backend is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"Backend connection failed: {str(e)}")
        return False

def main():
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    
    # Get current date for filename
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Modify results path to include date
    results_path = RESULTS_JSON_PATH.replace('.json', f'_{current_date}.json')
    
    # Clear the database
    print("Clearing database...")
    clear_database()
    
    # Upload the PDF
    print("Uploading PDF...")
    upload_pdf()
    
    # Wait a bit for indexing
    print("Waiting for indexing...")
    time.sleep(60)  
    
    # Read questions
    print("Reading questions...")
    questions = read_questions()
    
    # Query each question and collect results
    results = []
    total_questions = len(questions)
    
    print(f"Processing {total_questions} questions...")
    for i, q in enumerate(questions, 1):
        print(f"Processing question {i}/{total_questions}")
        answer = query_backend(q["question"])
        results.append({
            "date": current_date,
            "id": q["id"],
            "question": q["question"],
            "answer": answer
        })
    
    # Save results with date in filename
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation complete. Results saved to {results_path}")

    if not test_backend_connection():
        raise Exception("Backend server is not accessible")
        
    return results_path  # Return the path of the generated file

if __name__ == "__main__":
    main() 