import json
import csv
import requests
from pathlib import Path
import time
from datetime import datetime
from prepare_answers import main as prepare_answers_main

def get_phi_evaluation(question, answer, max_retries=3):
    """Query model to evaluate the RAG response with retry logic"""
    
    prompt = """You are an expert evaluator of RAG (Retrieval Augmented Generation) systems. Your task is to evaluate the following question and answer pair from a RAG system about the EU AI Act.

Please evaluate the response on a scale of 0-100 in the following categories:
1. Relevance: How well does the answer address the specific question asked?
2. Accuracy: How factually accurate is the information provided based on standard knowledge of the EU AI Act?
3. Completeness: How thorough and comprehensive is the answer?
4. Coherence: How well-structured and logically organized is the response?
5. Conciseness: How well does it balance detail with brevity?
6. Citation: How well does it references specific articles and sections?

For each category:
- Provide a score (0-100)
- Give a brief 1 sentence justification for the score

IMPORTANT: You must strictly follow this format for each category (one per line):
RELEVANCE|score|justification
ACCURACY|score|justification
COMPLETENESS|score|justification
COHERENCE|score|justification
CONCISENESS|score|justification
CITATION|score|justification

Question: {question}

Answer: {answer}

Remember: Your response must contain exactly 6 lines, each with the format CATEGORY|score|justification"""

    for attempt in range(max_retries):
        try:
            # Make request to Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "phi4:14b",
                    "prompt": prompt.format(question=question, answer=answer),
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                eval_text = response.json()["response"]
                
                # Validate response format
                lines = [line for line in eval_text.strip().split('\n') if '|' in line]
                if len(lines) == 6 and all(len(line.split('|')) == 3 for line in lines):
                    return eval_text
                else:
                    print(f"Attempt {attempt + 1}: Invalid response format, retrying...")
            else:
                print(f"Attempt {attempt + 1}: HTTP error {response.status_code}, retrying...")
            
            # Add increasing delay between retries
            time.sleep(2 ** attempt)
            
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error occurred: {str(e)}, retrying...")
            time.sleep(2 ** attempt)
    
    # If all retries failed, return a default evaluation
    return """RELEVANCE|50|Could not evaluate properly
ACCURACY|50|Could not evaluate properly
COMPLETENESS|50|Could not evaluate properly
COHERENCE|50|Could not evaluate properly
CONCISENESS|50|Could not evaluate properly
CITATION|50|Could not evaluate properly"""

def parse_evaluation(eval_text):
    """Parse the evaluation response into a dictionary with error handling"""
    results = {}
    try:
        lines = [line for line in eval_text.strip().split('\n') if '|' in line]
        for line in lines:
            parts = line.split('|')
            if len(parts) >= 3:  # Handle cases where justification might contain '|'
                category = parts[0]
                score = parts[1]
                justification = '|'.join(parts[2:])  # Rejoin any split justification
                try:
                    results[category.lower()] = {
                        'score': int(score),
                        'justification': justification.strip()
                    }
                except ValueError:
                    print(f"Warning: Could not parse score in line: {line}")
                    continue
    except Exception as e:
        print(f"Error parsing evaluation: {str(e)}")
    return results

def main():
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Get current date for filename
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Read the RAG results
    with open(f"eval/results/results_{current_date}.json", "r") as f:
        rag_results = json.load(f)
    
    # Prepare CSV output
    csv_rows = []
    csv_headers = ['question_id', 'question', 'relevance_score', 'relevance_justification',
                  'accuracy_score', 'accuracy_justification', 'completeness_score',
                  'completeness_justification', 'coherence_score', 'coherence_justification',
                  'conciseness_score', 'conciseness_justification', 'citation_score',
                  'citation_justification', 'average_score']
    
    # Process each question and answer
    total = len(rag_results)
    for i, result in enumerate(rag_results, 1):
        print(f"Evaluating response {i}/{total}")
        
        try:
            # Get evaluation from phi4
            eval_text = get_phi_evaluation(result["question"], result["answer"])
            eval_results = parse_evaluation(eval_text)
            
            # Print the evaluation scores
            print(eval_results)
            
            
            # Calculate average score
            scores = [v['score'] for v in eval_results.values()]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Prepare row for CSV
            row = {
                'question_id': result['id'],
                'question': result['question'],
                'average_score': avg_score
            }
            
            # Add individual category scores and justifications
            for category in ['relevance', 'accuracy', 'completeness', 'coherence', 'conciseness', 'citation']:
                if category in eval_results:
                    row[f'{category}_score'] = eval_results[category]['score']
                    row[f'{category}_justification'] = eval_results[category]['justification']
                else:
                    row[f'{category}_score'] = 0
                    row[f'{category}_justification'] = 'No evaluation provided'
            
            csv_rows.append(row)
            
        except Exception as e:
            print(f"Error processing response {i}: {str(e)}")
            # Add a row with error information
            row = {
                'question_id': result['id'],
                'question': result['question'],
                'average_score': 0
            }
            for category in ['relevance', 'accuracy', 'completeness', 'coherence', 'conciseness', 'citation']:
                row[f'{category}_score'] = 0
                row[f'{category}_justification'] = f'Error: {str(e)}'
            csv_rows.append(row)
        
        # Add a small delay between evaluations
        time.sleep(1)
    
    # Write results to CSV with date in filename
    output_file = f'eval/results/evaluation_{current_date}.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"Evaluation complete. Results saved to {output_file}")

def run_full_evaluation():
    """Run both prepare_answers and evaluation in sequence"""
    print("Step 1: Preparing answers...")
    prepare_answers_main()
    
    print("\nStep 2: Evaluating responses...")
    main()

if __name__ == "__main__":
    run_full_evaluation() 