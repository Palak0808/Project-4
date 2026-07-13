# CareerRAG - Intelligent Resume Match & Career Assistant

An AI-powered smart HR assessment dashboard that analyzes candidate resumes against job descriptions, identifies skill alignments, builds interactive monthly upskilling timelines, and runs Retrieval-Augmented Generation (RAG) queries over the combined context using local browser vector embeddings.

---

## 🚀 Web Application Live Deployment
CareerRAG is now fully modernized as a client-side AI web application! You can deploy it to **Vercel** with a single click.

* **Client-Side Embeddings:** Powered by Hugging Face Transformers.js executing `all-MiniLM-L6-v2` directly in the browser via ONNX Runtime Web.
* **Client-Side PDF Parsing:** Powered by PDF.js to extract raw texts from resume attachments directly in-browser.
* **Zero Backend Costs:** 100% serverless static deployment that requires no API keys or Python server instances.

---

## 🛠️ System Architecture & Features

CareerRAG consists of a dual implementation:
1. **Python Jupyter Notebook (`Career_RAG_Assistant.ipynb`)**: Original model developer version using `pypdf`, `sentence-transformers`, `faiss-cpu`, and `reportlab`.
2. **Client-Side Web Dashboard (`index.html`, `app.js`, `style.css`)**: High-performance responsive web dashboard deployable to static hosts like Vercel.

### Key Web Features:
* **PDF Resume Text Extraction**: Drag-and-drop a PDF resume. The browser extracts the raw text in real-time, showing a collapsible preview panel.
* **Job Description Templates**: Quickly load template JDs for **Machine Learning Engineer, Data Scientist, AI Engineer, and Backend Developer**, or type in a custom JD.
* **Dynamic Skills Gap Analyzer**: Identifies matching skills (with checkmarks) and missing skills (with warning labels) compared against JD prerequisites.
* **Interactive Upskilling Roadmap**: Automatically generates a month-by-month learning checklist for the missing skills, complete with mock guidelines and interactive checkboxes.
* **Role Fit Matcher**: Evaluates candidate fit across multiple target roles (Data Scientist, ML Engineer, etc.) based on skills intersections, and recommends the best-fitting role.
* **RAG Q&A Console**: A text input box allowing you to query details about the candidate (e.g. "What is the candidate's experience in AWS?"). The system embeds the query, searches in-memory vector embeddings (acting as a client-side FAISS FlatL2 index), and retrieves the top 3 relevant text passages with similarity percentages and source labels.
* **PDF Exporter**: Assembles a professional downloadable `Career_Assessment_Report.pdf` with the candidate details, match scores, missing skills lists, monthly roadmaps, and top retrieved RAG context.

---

## ⚙️ Running Locally

Since the web application is built with vanilla HTML, CSS, and JS (fetching assets from CDNs and importing ES modules), it requires **no local compilation or package installations**.

To run it locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/Palak0808/Project-4.git
   cd Project-4
   ```
2. Run a simple local HTTP server (required to allow ES modules and cache storage mechanisms):
   ```bash
   # Using Python 3
   python -m http.server 8000
   
   # Or using Node.js (if installed)
   npx serve .
   ```
3. Open `http://localhost:8000` in your web browser.

---

## ⚡ Deploying to Vercel

To host this project on Vercel:
1. Push this project to your GitHub repository.
2. Go to the [Vercel Dashboard](https://vercel.com/dashboard) and click **Add New > Project**.
3. Import the `Project-4` repository.
4. Keep the default settings (Framework Preset: **Other**, Build Command: **None**, Output Directory: **Current Directory / Root**).
5. Click **Deploy**. Vercel will host your static website within seconds!

---

## 🧪 Original Python Notebook Methodology

The research implementation uses:
* **SentenceTransformer ("all-MiniLM-L6-v2")** to calculate text embeddings.
* **FAISS IndexFlatL2** to index and retrieve nearest neighbor document chunks.
* **ReportLab Canvas** to export PDF report structures.

The original code is preserved in [Career_RAG_Assistant.ipynb](file:///C:/Users/palak/.gemini/antigravity-ide/scratch/Project-4/Career_RAG_Assistant.ipynb).