# This is the main Flask application for the resume screener.
# It includes logic to download model files from external storage during deployment.

from flask import Flask, render_template_string, request, jsonify
import re
import json
import numpy as np
import pickle
import os
import requests
import shutil

app = Flask(__name__)

# --- 1. Configuration & Model Paths ---

# !!! CRITICAL: REPLACE THESE PLACEHOLDERS WITH YOUR ACTUAL DIRECT DOWNLOAD URLS !!!
DOWNLOAD_URLS = {
    'tfidf.pkl': 'https://drive.google.com/uc?export=download&id=1g80R40ll8uhc917KzPZUAIlcEjINupbK',
    'clf.pkl': 'https://drive.google.com/uc?export=download&id=1fwD_lSAK7THYD4BIR1SDMTN675GdDus-',
    'encoder.pkl': 'https://drive.google.com/uc?export=download&id=18d8CcQBR3fYjuP_S43GovJgfQti4wAq5',
}

# Define paths relative to this script
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
CATEGORIES_FILE = os.path.join(os.path.dirname(__file__), 'resume_categories.json')

# Global variables for model components
tfidf = None
svc_model = None
le = None
CATEGORY_MAP = None
INVERSE_TRANSFORM = {}
SUGGESTIONS_MAP = {}
MODELS_LOADED_SUCCESSFULLY = False

# --- 2. Model Download and Loading ---

def download_file(url, local_filename):
    """Downloads a file from a URL to a local path."""
    print(f"INFO: Starting download of {os.path.basename(local_filename)}...")
    if url.startswith('YOUR_'):
        raise ValueError(f"URL for {os.path.basename(local_filename)} is a placeholder. Please update DOWNLOAD_URLS.")
    
    # Use stream=True for large files
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print(f"INFO: Downloaded {os.path.basename(local_filename)} successfully.")

def load_models():
    """Loads or downloads and loads the pickled model objects and category data."""
    global tfidf, svc_model, le, CATEGORY_MAP, INVERSE_TRANSFORM, SUGGESTIONS_MAP, MODELS_LOADED_SUCCESSFULLY
    
    if MODELS_LOADED_SUCCESSFULLY:
        return

    try:
        # 1. Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"INFO: Model directory check: {MODEL_DIR}")

        # 2. Check if models exist. If not, download them.
        for filename, url in DOWNLOAD_URLS.items():
            filepath = os.path.join(MODEL_DIR, filename)
            if not os.path.exists(filepath):
                download_file(url, filepath)

        # 3. Load ML components
        with open(os.path.join(MODEL_DIR, 'tfidf.pkl'), 'rb') as f:
            tfidf = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'clf.pkl'), 'rb') as f:
            svc_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'encoder.pkl'), 'rb') as f:
            le = pickle.load(f)

        # 4. Load Category Mapping
        with open(CATEGORIES_FILE, 'r') as f:
            CATEGORY_MAP = json.load(f)
            
        # 5. Create lookup dictionaries
        category_names = le.classes_
        INVERSE_TRANSFORM = {i: name for i, name in enumerate(category_names)}
        SUGGESTIONS_MAP = {data['name']: data['suggestion'] for data in CATEGORY_MAP}
        
        print("All models and data loaded successfully!")
        MODELS_LOADED_SUCCESSFULLY = True
        
    except Exception as e:
        # This will be printed to the Render logs if loading fails
        print(f"CRITICAL ERROR: Failed to load or download model files. Prediction will be disabled. Details: {e}")
        MODELS_LOADED_SUCCESSFULLY = False


# Load models when the app starts
load_models()

# --- 3. Utility Functions ---

def cleanResume(txt):
    """
    Cleans the resume text using raw strings (r'...') to avoid SyntaxWarnings 
    and handles file-read artifacts.
    """
    cleanText = txt.strip()
    # Use raw strings to fix SyntaxWarning
    cleanText = re.sub(r'http\S+\s', ' ', cleanText)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)  
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', ' ', cleanText)
    
    # Remove non-ASCII characters and clean up excessive whitespace
    cleanText = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleanText) 
    
    # Replace multiple spaces and newlines with a single space
    cleanText = re.sub(r'\s+', ' ', cleanText)
    
    return cleanText.strip()

def predict_category_and_suggest(input_resume):
    """Predicts the category and retrieves suggestions using the loaded model."""
    global MODELS_LOADED_SUCCESSFULLY
    
    if not MODELS_LOADED_SUCCESSFULLY:
        # Return a custom error message if models are not loaded
        return "Model Unavailable", "Please verify your model download URLs and check the Render service logs for download errors."

    # 1. Clean
    cleaned_text = cleanResume(input_resume)

    # 2. Vectorize (tfidf.transform returns a sparse matrix)
    vectorized_text = tfidf.transform([cleaned_text])
    
    # Ensure the model receives the correct array format
    try:
        vectorized_text_dense = vectorized_text.toarray()
    except:
        vectorized_text_dense = vectorized_text
        
    # 3. Predict (returns array, e.g., [10])
    # The [0] is safe because we only pass one resume string
    predicted_category_index_encoded = svc_model.predict(vectorized_text_dense)[0]
    
    # 4. Decode
    predicted_category_name = le.inverse_transform([predicted_category_index_encoded])[0]

    # 5. Get Suggestion
    suggestion = SUGGESTIONS_MAP.get(predicted_category_name, "No specific suggestions available. Focus on quantifying achievements and listing key skills clearly.")

    return predicted_category_name, suggestion

# --- 4. Flask Application Setup (HTML is embedded) ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Resume Screener Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f4f7f9; }
        .card { box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06); }
        .file-upload-box {
            border: 2px dashed #cbd5e1;
            transition: border-color 0.15s ease-in-out;
        }
        .file-upload-box:hover {
            border-color: #a3bffa; /* indigo-300 */
        }
    </style>
</head>
<body class="min-h-screen p-4 sm:p-8">

    <!-- Main Content Card -->
    <div class="max-w-4xl mx-auto bg-white p-6 sm:p-10 rounded-xl card">
        <h1 class="text-4xl font-extrabold text-gray-900 mb-2 text-center">AI Resume Screener</h1>
        <p class="text-center text-gray-500 mb-8">Paste your resume text or upload a file for category classification and tailored suggestions.</p>

        <!-- Upload/Paste Toggle -->
        <div class="mb-6 flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4">
            
            <div class="w-full sm:w-1/2">
                <label for="resume-file" class="block text-sm font-medium text-gray-700 mb-2">1. Upload Resume File (TXT, PDF):</label>
                <div class="file-upload-box relative p-4 rounded-lg bg-gray-50 flex justify-center items-center h-20">
                    <input type="file" id="resume-file" accept=".txt,.pdf" class="absolute inset-0 opacity-0 cursor-pointer">
                    <p class="text-gray-500 text-sm text-center">
                        Click here to browse or drag & drop a file.
                    </p>
                    <div id="file-status" class="absolute top-0 right-0 p-1 text-xs font-semibold rounded-bl-lg"></div>
                </div>
                <p class="text-xs text-gray-500 mt-2">
                    Note: PDF upload is simulated; for real use, upload a plain **.txt** file or set up a server-side PDF parser.
                </p>
            </div>

            <div class="w-full sm:w-1/2">
                <label for="resume-text" class="block text-sm font-medium text-gray-700 mb-2">2. Or Paste Resume Text:</label>
                <textarea id="resume-text" rows="8" class="w-full p-4 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 transition duration-150 ease-in-out resize-none" placeholder="Paste the raw text content of your resume..."></textarea>
            </div>
        </div>

        <div class="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-4">
            <button id="predict-btn" class="w-full sm:w-auto px-6 py-3 text-lg font-semibold rounded-lg text-white bg-indigo-600 hover:bg-indigo-700 transition duration-150 ease-in-out focus:outline-none focus:ring-4 focus:ring-indigo-500 focus:ring-opacity-50">
                Analyze Resume
            </button>
            <button id="clear-btn" class="w-full sm:w-auto px-6 py-3 text-lg font-semibold rounded-lg text-indigo-700 bg-indigo-100 hover:bg-indigo-200 transition duration-150 ease-in-out focus:outline-none focus:ring-4 focus:ring-indigo-500 focus:ring-opacity-50">
                Clear All
            </button>
        </div>

        <!-- Result Section (Hidden by Default) -->
        <div id="results-container" class="mt-10 p-6 bg-gray-50 border border-gray-200 rounded-lg hidden">
            <h2 class="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                <svg class="w-6 h-6 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944v0m.715 14.819l1.458 2.379a11.9 11.9 0 005.13-7.563m-.706-5.266A11.95 11.95 0 0012 2.944c-1.472 0-2.864.33-4.142.948m-5.13 7.563a11.9 11.9 0 005.13 7.563l1.458-2.379m.715-14.819C6.467 4.717 5 7.94 5 12c0 3.863 1.258 7.234 3.39 9.996"></path></svg>
                Analysis Results
            </h2>
            
            <div class="mb-4 p-4 bg-indigo-50 rounded-lg border-l-4 border-indigo-500">
                <p class="text-sm font-medium text-indigo-800">Predicted Job Category:</p>
                <p id="predicted-category" class="text-3xl font-extrabold text-indigo-600 mt-1">...</p>
            </div>

            <div class="p-4 bg-yellow-50 rounded-lg border-l-4 border-yellow-500">
                <p class="text-sm font-medium text-yellow-800">Suggestions for Improvement:</p>
                <p id="suggestions" class="text-base text-gray-700 mt-1">...</p>
            </div>
        </div>
        
        <!-- Loading Indicator -->
        <div id="loading-indicator" class="mt-10 flex justify-center items-center hidden">
            <svg class="animate-spin -ml-1 mr-3 h-8 w-8 text-indigo-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span class="text-indigo-600 font-medium">Analyzing resume data...</span>
        </div>

        <!-- System Status/Error Message -->
        <div id="status-message" class="mt-4 text-center text-red-600 font-semibold hidden"></div>
        
        <!-- Footer Note -->
        <p class="mt-8 text-xs text-gray-400 text-center">
            This application uses your trained SVC model and TF-IDF vectorizer for classification.
        </p>

    </div>

    <script>
        
        const resumeFileEl = document.getElementById('resume-file');
        const resumeTextEl = document.getElementById('resume-text');
        const fileStatusEl = document.getElementById('file-status');
        const resultsContainer = document.getElementById('results-container');
        const loadingIndicator = document.getElementById('loading-indicator');
        const statusMessage = document.getElementById('status-message');

        // Check for Model Loading Status on page load
        (async function checkModelStatus() {
            try {
                // A simple check to see if the server returns the model unavailable error
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ resume: 'health check' })
                });
                const data = await response.json();
                
                if (data.category === "Model Unavailable" || data.error && data.error.includes("Model Unavailable")) {
                    statusMessage.textContent = "WARNING: Model files failed to load on the server. Prediction is currently disabled. Check Render logs.";
                    statusMessage.classList.remove('hidden');
                    // Disable prediction button visually
                    document.getElementById('predict-btn').disabled = true;
                    document.getElementById('predict-btn').classList.add('opacity-50', 'cursor-not-allowed');

                }
            } catch (e) {
                // Ignore network errors during initial health check
            }
        })();


        function setFileStatus(text, color) {
            fileStatusEl.textContent = text;
            fileStatusEl.className = `absolute top-0 right-0 p-1 text-xs font-semibold rounded-bl-lg ${color}`;
        }
        
        // --- File Reader ---
        function readFile(file) {
            return new Promise((resolve, reject) => {
                const fileType = file.name.split('.').pop().toLowerCase();

                if (fileType !== 'txt' && fileType !== 'pdf') {
                     reject(new Error("Unsupported file type. Please upload .txt or .pdf."));
                     return;
                }

                if (fileType === 'pdf') {
                    // NOTE: Real PDF parsing is complex. We simulate success here.
                    setFileStatus('Parsing PDF (Simulated)...', 'bg-yellow-200 text-yellow-800');
                    setTimeout(() => {
                        // Using a generic, clean text string for simulation that should predict Data Science
                        const mockText = "John Doe. Experience in Python, Machine Learning, and Data Analysis. 2020-Present: Data Scientist at TechCorp. Projects include NLP for sentiment analysis and CNN for image classification.";
                        resolve(mockText);
                    }, 500);
                    return;
                }
                
                const reader = new FileReader();
                reader.onload = (e) => resolve(e.target.result);
                reader.onerror = (e) => reject(e);
                
                // Read as text for TXT files (using UTF-8 encoding)
                reader.readAsText(file, 'UTF-8');
            });
        }
        
        // --- Event Listeners ---

        // 1. File Input Change Listener
        resumeFileEl.addEventListener('change', async (event) => {
            const file = event.target.files[0];
            if (!file) {
                setFileStatus('', '');
                return;
            }

            // Clear previous results and loading indicators
            resultsContainer.classList.add('hidden');
            loadingIndicator.classList.remove('hidden');
            statusMessage.classList.add('hidden');
            resumeTextEl.value = ''; // Clear text area before loading

            try {
                // Check if file size is reasonable (e.g., max 2MB for a quick check)
                if (file.size > 1024 * 1024 * 2) { 
                    setFileStatus('File Too Large', 'bg-red-200 text-red-800');
                    statusMessage.textContent = "Error: File size exceeds the 2MB limit for client-side reading.";
                    statusMessage.classList.remove('hidden');
                    return;
                }
                
                const textContent = await readFile(file);
                resumeTextEl.value = textContent;
                
                // Immediately run prediction on load, which is a common pattern
                document.getElementById('predict-btn').click(); 

            } catch (e) {
                console.error("File Read Error:", e);
                setFileStatus('Error', 'bg-red-200 text-red-800');
                statusMessage.textContent = e.message || `Error: Could not read file ${file.name}.`;
                statusMessage.classList.remove('hidden');
            } finally {
                loadingIndicator.classList.add('hidden');
            }
        });
        
        // 2. Predict Button Listener
        document.getElementById('predict-btn').addEventListener('click', async () => {
            const resumeText = resumeTextEl.value;
            const predictedCategoryEl = document.getElementById('predicted-category');
            const suggestionsEl = document.getElementById('suggestions');
            
            statusMessage.classList.add('hidden');

            if (!resumeText.trim()) {
                predictedCategoryEl.textContent = "Input Required";
                suggestionsEl.textContent = "Please paste or upload a resume to analyze.";
                resultsContainer.classList.remove('hidden');
                return;
            }

            resultsContainer.classList.add('hidden');
            loadingIndicator.classList.remove('hidden');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ resume: resumeText })
                });

                const data = await response.json();

                if (response.ok) {
                    predictedCategoryEl.textContent = data.category;
                    suggestionsEl.textContent = data.suggestion;
                } else {
                    // Display server-side error message
                    statusMessage.textContent = data.error || "An unknown server error occurred.";
                    statusMessage.classList.remove('hidden');
                    predictedCategoryEl.textContent = data.category || "Analysis Failed (Error)";
                    suggestionsEl.textContent = data.suggestion || "Check the console or status message above for server details.";
                }

            } catch (error) {
                console.error('Fetch error:', error);
                statusMessage.textContent = "Network Error: Could not connect to the server. Is Flask running?";
                statusMessage.classList.remove('hidden');
                predictedCategoryEl.textContent = "System Error";
                suggestionsEl.textContent = "Check if the Flask server is running correctly.";
            } finally {
                loadingIndicator.classList.add('hidden');
                resultsContainer.classList.remove('hidden');
            }
        });
        
        // 3. Clear Button Listener
        document.getElementById('clear-btn').addEventListener('click', () => {
            resumeTextEl.value = '';
            resumeFileEl.value = ''; // Clear file input cache
            setFileStatus('', '');
            resultsContainer.classList.add('hidden');
            statusMessage.classList.add('hidden');
            
            // Re-enable button if it was disabled
            document.getElementById('predict-btn').disabled = false;
            document.getElementById('predict-btn').classList.remove('opacity-50', 'cursor-not-allowed');
        });
    </script>

</body>
</html>
"""

# Flask Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    resume_text = data.get('resume', '')

    # Health check for model status on client side
    if resume_text == 'health check':
        if MODELS_LOADED_SUCCESSFULLY:
            return jsonify({"category": "Model Available", "suggestion": "System operational."})
        else:
            # Explicitly return Model Unavailable status
            return jsonify({"category": "Model Unavailable", "suggestion": "Model loading failed on server."}), 503

    if not resume_text:
        return jsonify({"error": "No resume text provided."}), 400

    try:
        category, suggestion = predict_category_and_suggest(resume_text)
        
        # Check if the function returned the error state
        if category == "Model Unavailable":
            return jsonify({"category": category, "suggestion": suggestion, "error": "Model Unavailable"}), 503

        return jsonify({
            "category": category,
            "suggestion": suggestion
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": "An internal error occurred during prediction."}), 500

# Entry point for the file (for local testing)
if __name__ == '__main__':
    # Flask application runs via gunicorn/flask run on Render
    # This block is typically only executed when running python app.py locally
    # It ensures the model loading happens regardless of the server method.
    pass
