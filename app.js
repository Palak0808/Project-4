// CareerRAG - Intelligent Resume Match & Career Assistant App Logic
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Configure transformers.js to load models from Hugging Face CDN (client-side only)
env.allowLocalModels = false;

// Global Application State
let extractor = null;
let modelLoading = true;
let resumeRawText = '';
let jdRawText = '';
let resumeFileName = '';
let isAnalyzing = false;

// Vector Database Chunks State
let chunksDB = []; // Array of { text, source, embedding }

// Define Job Description templates
const jdTemplates = {
    'machine-learning-engineer': `Machine Learning Engineer

Required Skills:
Python
Machine Learning
TensorFlow
Docker
AWS
SQL

Responsibilities:
Develop machine learning models.
Deploy AI applications.
Work with cloud technologies.`,

    'data-scientist': `Data Scientist

Required Skills:
Python
SQL
Machine Learning
Pandas
Scikit-Learn
Statistics

Responsibilities:
Analyze large datasets.
Build predictive statistical models.
Communicate insights via data visualization.`,

    'ai-engineer': `AI Engineer

Required Skills:
Python
TensorFlow
AWS
NLP
Generative AI
PyTorch

Responsibilities:
Design and deploy neural networks.
Integrate LLM API gateways.
Maintain cloud infrastructure.`,

    'backend-developer': `Backend Developer

Required Skills:
Python
SQL
Docker
Django
REST APIs
Git

Responsibilities:
Build and maintain web backend servers.
Optimize relational database queries.
Configure Docker containers for staging environments.`
};

// Target skills mapping across all roles
const allKnownSkills = [
    "python",
    "machine learning",
    "tensorflow",
    "docker",
    "aws",
    "sql",
    "pandas",
    "scikit-learn",
    "statistics",
    "nlp",
    "generative ai",
    "pytorch",
    "django",
    "rest apis",
    "git"
];

// Target roles definition
const targetRoles = {
    "Data Scientist": ["python", "sql", "machine learning"],
    "Machine Learning Engineer": ["python", "machine learning", "tensorflow", "docker"],
    "AI Engineer": ["python", "tensorflow", "aws"],
    "Backend Developer": ["python", "sql"]
};

// PDF.js configuration
const pdfjsLib = window['pdfjs-dist/build/pdf'];
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js';

// UI Elements Selection
const elements = {
    engineStatusBadge: document.getElementById('engine-status-badge'),
    modelProgressContainer: document.getElementById('model-progress-container'),
    modelProgressLabel: document.getElementById('model-progress-label'),
    modelProgressBar: document.getElementById('model-progress-bar'),
    
    btnReset: document.getElementById('btn-reset'),
    btnDownloadPdf: document.getElementById('btn-download-pdf'),
    btnAnalyze: document.getElementById('btn-analyze'),
    
    // Resume uploader elements
    uploadZone: document.getElementById('upload-zone'),
    fileInput: document.getElementById('file-input'),
    fileStatusBox: document.getElementById('file-status-box'),
    lblFileName: document.getElementById('lbl-file-name'),
    lblFileSize: document.getElementById('lbl-file-size'),
    lblFilePages: document.getElementById('lbl-file-pages'),
    
    textPreviewContainer: document.getElementById('text-preview-container'),
    extractedTextPreview: document.getElementById('extracted-text-preview'),
    
    // Job description elements
    jdTemplateSelect: document.getElementById('jd-template-select'),
    textareaJd: document.getElementById('textarea-jd'),
    
    // Telemetry dashboard elements
    lblMatchScore: document.getElementById('lbl-match-score'),
    lblRecRole: document.getElementById('lbl-rec-role'),
    lblSkillsFoundCount: document.getElementById('lbl-skills-found-count'),
    lblSkillsMissingCount: document.getElementById('lbl-skills-missing-count'),
    scoreRing: document.getElementById('score-ring'),
    
    containerSkillsFound: document.getElementById('container-skills-found'),
    containerSkillsMissing: document.getElementById('container-skills-missing'),
    
    // Roles list bars
    roleFitBars: document.getElementById('role-fit-bars'),
    
    // Upskilling Timeline roadmap
    roadmapTimelineContainer: document.getElementById('roadmap-timeline-container'),
    
    // RAG Console elements
    inputQuery: document.getElementById('input-query'),
    btnQuery: document.getElementById('btn-query'),
    lblVectorCount: document.getElementById('lbl-vector-count'),
    qaResultsContainer: document.getElementById('qa-results-container'),
    
    // Nav buttons
    btnDashboard: document.getElementById('btn-dashboard'),
    btnQaNav: document.getElementById('btn-qa-nav'),
    btnRoadmapNav: document.getElementById('btn-roadmap-nav')
};

// Initialize Application
window.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    loadJdTemplate();
    await loadEmbeddingModel();
});

// 1. Load Transformers.js Embedding Model
async function loadEmbeddingModel() {
    try {
        updateEngineStatus('Loading Model...', 'loading');
        elements.modelProgressContainer.classList.remove('hidden');
        
        extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
            progress_callback: (data) => {
                if (data.status === 'progress') {
                    const percent = Math.round(data.progress);
                    elements.modelProgressBar.style.width = `${percent}%`;
                    elements.modelProgressLabel.textContent = `Downloading: ${percent}%`;
                } else if (data.status === 'ready') {
                    elements.modelProgressBar.style.width = '100%';
                    elements.modelProgressLabel.textContent = 'Preparing model...';
                }
            }
        });
        
        elements.modelProgressContainer.classList.add('hidden');
        modelLoading = false;
        updateEngineStatus('Engine Ready', 'ready');
        enableInputsIfReady();
    } catch (error) {
        console.error('Error loading semantic embedding model:', error);
        elements.modelProgressLabel.textContent = 'Loading failed.';
        updateEngineStatus('Engine Error', 'loading');
        alert('Failed to load semantic embedding model. Please check internet connection.');
    }
}

function updateEngineStatus(text, stateClass) {
    elements.engineStatusBadge.className = `status-indicator ${stateClass}`;
    elements.engineStatusBadge.querySelector('.status-text').textContent = text;
}

// 2. Setup Event Listeners
function setupEventListeners() {
    // JD selector changes
    elements.jdTemplateSelect.addEventListener('change', loadJdTemplate);
    
    // File inputs
    elements.uploadZone.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Drag & Drop
    elements.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadZone.classList.add('dragover');
    });
    
    elements.uploadZone.addEventListener('dragleave', () => {
        elements.uploadZone.classList.remove('dragover');
    });
    
    elements.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            processResumeFile(e.dataTransfer.files[0]);
        }
    });

    // Assessment triggers
    elements.btnAnalyze.addEventListener('click', runSemanticAssessment);
    elements.btnQuery.addEventListener('click', askRAGAssistant);
    elements.inputQuery.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') askRAGAssistant();
    });

    // Sidebar & action triggers
    elements.btnReset.addEventListener('click', resetSystem);
    elements.btnDownloadPdf.addEventListener('click', generateAssessmentReport);
}

// Enable analyze button if model loaded and resume loaded
function enableInputsIfReady() {
    if (!modelLoading && resumeRawText.trim().length > 0) {
        elements.btnAnalyze.removeAttribute('disabled');
    } else {
        elements.btnAnalyze.setAttribute('disabled', 'true');
    }
}

// 3. Load Job Description Template
function loadJdTemplate() {
    const selected = elements.jdTemplateSelect.value;
    if (selected in jdTemplates) {
        elements.textareaJd.value = jdTemplates[selected];
        elements.textareaJd.removeAttribute('disabled');
    } else {
        elements.textareaJd.value = '';
        elements.textareaJd.focus();
    }
}

// 4. Extract Text from PDF Resume
function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        processResumeFile(e.target.files[0]);
    }
}

async function processResumeFile(file) {
    if (file.type !== 'application/pdf') {
        alert('Please upload a PDF format resume.');
        return;
    }
    
    resumeFileName = file.name;
    elements.lblFileName.textContent = file.name;
    elements.lblFileSize.textContent = `${Math.round(file.size / 1024)} KB`;
    elements.fileStatusBox.classList.remove('hidden');
    
    // Change upload zone styling to success/completed state
    elements.resumeUploadTitle.textContent = "PDF Uploaded successfully";
    elements.resumeUploadDesc.textContent = file.name;
    elements.uploadZone.style.borderColor = "var(--green)";
    
    // Show spinner overlay in preview box
    elements.textPreviewContainer.style.display = 'block';
    elements.extractedTextPreview.innerHTML = 'Extracting resume text...';
    
    try {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        elements.lblFilePages.textContent = `${pdf.numPages} ${pdf.numPages === 1 ? 'page' : 'pages'}`;
        
        let text = '';
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            const pageText = content.items.map(item => item.str).join(' ');
            text += pageText + '\n';
        }
        
        resumeRawText = text;
        elements.extractedTextPreview.textContent = text;
        
        enableInputsIfReady();
    } catch (error) {
        console.error('PDF text extraction error:', error);
        elements.extractedTextPreview.textContent = 'Failed to extract text. Make sure the file is not encrypted.';
        alert('Error extracting text from PDF.');
    }
}

// 5. Text Chunking Utility (Matches 100 words sliding window in Python Notebook)
function createChunks(text) {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const chunks = [];
    const chunkSize = 100;
    
    for (let i = 0; i < words.length; i += chunkSize) {
        const chunk = words.slice(i, i + chunkSize).join(' ');
        chunks.push(chunk);
    }
    
    return chunks;
}

// 6. Vector Embeddings Generation (Semantic Assessment)
async function runSemanticAssessment() {
    if (isAnalyzing || modelLoading || !resumeRawText) return;
    
    isAnalyzing = true;
    elements.btnAnalyze.innerHTML = '<i class="spinner" style="width:16px;height:16px;border-width:2px;margin:0"></i> Analyzing Vectors...';
    elements.btnAnalyze.setAttribute('disabled', 'true');
    
    // Yield threat execution to allow UI loaders
    await new Promise(r => setTimeout(r, 50));
    
    try {
        jdRawText = elements.textareaJd.value;
        
        // Chunk both texts
        const resumeChunks = createChunks(resumeRawText);
        const jdChunks = createChunks(jdRawText);
        
        // Clear old database
        chunksDB = [];
        elements.lblVectorCount.textContent = 'Processing...';
        
        // Encode Resume Chunks
        for (let i = 0; i < resumeChunks.length; i++) {
            const chunk = resumeChunks[i];
            const embedding = await getEmbedding(chunk);
            chunksDB.push({
                text: chunk,
                source: 'Resume',
                embedding: embedding
            });
        }
        
        // Encode JD Chunks
        for (let i = 0; i < jdChunks.length; i++) {
            const chunk = jdChunks[i];
            const embedding = await getEmbedding(chunk);
            chunksDB.push({
                text: chunk,
                source: 'Job Description',
                embedding: embedding
            });
        }
        
        // Enable RAG Controls
        elements.lblVectorCount.textContent = `${chunksDB.length} Vectors`;
        elements.inputQuery.removeAttribute('disabled');
        elements.btnQuery.removeAttribute('disabled');
        
        // Clear Q&A console logs
        elements.qaResultsContainer.innerHTML = `
            <div class="empty-qa-placeholder" style="margin:auto;">
                <i data-lucide="sparkles" style="color:var(--accent-light)"></i>
                <p>Vector database created successfully! Ask any details about the candidate above.</p>
            </div>
        `;
        lucide.createIcons();
        
        // Run Skills analysis
        performSkillsAssessment();
        
        // Enable PDF Report Downloads
        elements.btnDownloadPdf.removeAttribute('disabled');
        
    } catch (error) {
        console.error('Assessment execution error:', error);
        alert('An error occurred during semantic assessment.');
    } finally {
        isAnalyzing = false;
        elements.btnAnalyze.innerHTML = '<i data-lucide="fingerprint"></i> Run Semantic Assessment';
        elements.btnAnalyze.removeAttribute('disabled');
        lucide.createIcons();
    }
}

// Generate text embedding using Xenova Transformers.js
async function getEmbedding(text) {
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

// 7. Skills Assessment & Recommendations Logic
function performSkillsAssessment() {
    // Parse JD required skills
    const jdSkills = allKnownSkills.filter(skill => jdRawText.toLowerCase().includes(skill));
    
    // Parse resume matched skills
    const resumeSkills = allKnownSkills.filter(skill => resumeRawText.toLowerCase().includes(skill));
    
    // Calculate found skills in JD requirements
    const foundSkills = jdSkills.filter(skill => resumeSkills.includes(skill));
    
    // Calculate missing skills in JD requirements
    const missingSkills = jdSkills.filter(skill => !resumeSkills.includes(skill));
    
    // Match Score: ((len(skills) - len(missing)) / len(skills)) * 100
    const matchScore = jdSkills.length > 0 
        ? ((jdSkills.length - missingSkills.length) / jdSkills.length) * 100 
        : 100;
        
    // Render Radial Score gauge
    animateScoreGauge(matchScore);
    
    // Render Stats
    elements.lblSkillsFoundCount.textContent = foundSkills.length;
    elements.lblSkillsMissingCount.textContent = missingSkills.length;
    
    // Render Tag Containers
    elements.containerSkillsFound.innerHTML = '';
    if (foundSkills.length === 0) {
        elements.containerSkillsFound.innerHTML = '<span class="empty-placeholder">No matching skills detected</span>';
    } else {
        foundSkills.forEach(skill => {
            const span = document.createElement('span');
            span.className = 'skill-tag found';
            span.textContent = skill.toUpperCase();
            elements.containerSkillsFound.appendChild(span);
        });
    }
    
    elements.containerSkillsMissing.innerHTML = '';
    if (missingSkills.length === 0) {
        elements.containerSkillsMissing.innerHTML = '<span class="empty-placeholder">No missing skills required! Excellent match!</span>';
    } else {
        missingSkills.forEach(skill => {
            const span = document.createElement('span');
            span.className = 'skill-tag missing';
            span.textContent = skill.toUpperCase();
            elements.containerSkillsMissing.appendChild(span);
        });
    }
    
    // Role Recommendation Breakdown
    const roleMatches = {};
    Object.entries(targetRoles).forEach(([roleName, roleSkills]) => {
        // Calculate intersection of candidate's found resume skills and this specific role's skills
        const intersection = roleSkills.filter(skill => resumeSkills.includes(skill));
        const fitScore = (intersection.length / roleSkills.length) * 100;
        roleMatches[roleName] = fitScore;
    });
    
    // Sort and get best role recommendation matching Notebook logic
    const bestRole = Object.keys(roleMatches).reduce((a, b) => roleMatches[a] >= roleMatches[b] ? a : b);
    elements.lblRecRole.textContent = `${bestRole}`;
    
    // Render role breakdown progress bars
    elements.roleFitBars.innerHTML = '';
    Object.entries(roleMatches).forEach(([roleName, fitScore]) => {
        const item = document.createElement('div');
        item.className = 'role-item';
        if (roleName === bestRole) {
            item.style.borderColor = 'rgba(16, 185, 129, 0.2)';
            item.style.background = 'rgba(16, 185, 129, 0.01)';
        }
        
        item.innerHTML = `
            <div class="role-meta">
                <span>${roleName}</span>
                <span class="role-pct" style="${roleName === bestRole ? 'color:var(--green)' : ''}">${fitScore.toFixed(0)}%</span>
            </div>
            <div class="progress-track">
                <div class="progress-bar" style="width: ${fitScore}%; ${roleName === bestRole ? 'background:var(--green)' : ''}"></div>
            </div>
        `;
        elements.roleFitBars.appendChild(item);
    });
    
    // Generate Personalized Learning Roadmap
    generateUpskillingTimeline(missingSkills);
}

function animateScoreGauge(targetScore) {
    const circle = elements.scoreRing;
    const r = circle.r.baseVal.value;
    const circumference = 2 * Math.PI * r;
    
    circle.style.strokeDasharray = circumference;
    
    // Update numeric text with counting animation
    let currentScore = 0;
    const increment = targetScore / 25;
    
    const countTimer = setInterval(() => {
        currentScore += increment;
        if (currentScore >= targetScore) {
            currentScore = targetScore;
            clearInterval(countTimer);
        }
        
        const formattedVal = Math.round(currentScore);
        elements.lblMatchScore.textContent = `${formattedVal}%`;
        
        // Update stroke-dashoffset
        const offset = circumference - (currentScore / 100) * circumference;
        circle.style.strokeDashoffset = offset;
        
        // Update stroke colors based on score value
        if (targetScore < 40) {
            circle.style.stroke = "var(--red)";
        } else if (targetScore < 70) {
            circle.style.stroke = "var(--yellow)";
        } else {
            circle.style.stroke = "var(--green)";
        }
    }, 20);
}

// Generate ups-killing learning timeline
function generateUpskillingTimeline(missingSkills) {
    elements.roadmapTimelineContainer.innerHTML = '';
    
    if (missingSkills.length === 0) {
        elements.roadmapTimelineContainer.innerHTML = `
            <div class="empty-timeline-placeholder">
                <i data-lucide="shield-check" style="color:var(--green)"></i>
                <h4 style="color:white;margin-top:0.5rem">Ready to Apply</h4>
                <p>The candidate matches all required skills for this job description!</p>
            </div>
        `;
        lucide.createIcons();
        return;
    }
    
    missingSkills.forEach((skill, index) => {
        const item = document.createElement('div');
        item.className = 'timeline-item';
        
        // Mock timelines and resource links matching skills
        const monthNum = index + 1;
        let timelineRec = '';
        if (skill === 'sql') {
            timelineRec = 'Master SQL join syntax, analytical windows functions, and query optimization techniques via PostgreSQL documentation and interactive tutorials.';
        } else if (skill === 'tensorflow' || skill === 'pytorch') {
            timelineRec = `Build neural network architectures, train classifiers, and optimize tensors. Check out the official ${skill.charAt(0).toUpperCase() + skill.slice(1)} developer certification modules.`;
        } else if (skill === 'docker') {
            timelineRec = 'Understand containerization concepts, write custom Dockerfiles, manage multi-container systems using Docker-Compose, and practice local registry set ups.';
        } else if (skill === 'aws') {
            timelineRec = 'Familiarize with cloud hosting, deploy static and docker pipelines using AWS EC2, S3 bucket storage policies, and configure Identity Access Management roles.';
        } else if (skill === 'machine learning' || skill === 'scikit-learn') {
            timelineRec = 'Study statistical ML estimators: linear/logistic regressions, random forests, clustering techniques, hyperparameter tunings, and model evaluation metrics.';
        } else {
            timelineRec = `Comprehensive study of ${skill.toUpperCase()}. Implement mini-projects on local environments and check public documentation references.`;
        }
        
        item.innerHTML = `
            <div class="timeline-dot"></div>
            <input type="checkbox" class="timeline-checkbox">
            <div class="timeline-item-content">
                <span class="timeline-month">Month ${monthNum} Plan</span>
                <span class="timeline-skill">Learn ${skill.toUpperCase()}</span>
                <p class="timeline-rec">${timelineRec}</p>
            </div>
        `;
        elements.roadmapTimelineContainer.appendChild(item);
    });
}

// 8. RAG QA Assistant (Similarity Search over Resume & JD chunks)
async function askRAGAssistant() {
    const query = elements.inputQuery.value.trim();
    if (!query || isAnalyzing) return;
    
    // Clear console input
    elements.inputQuery.value = '';
    
    // Append User Query Bubble
    const userBubble = document.createElement('div');
    userBubble.className = 'qa-bubble';
    userBubble.innerHTML = `
        <div class="qa-query">
            <span>${query}</span>
        </div>
    `;
    
    // Clear placeholder if first entry
    const placeholder = elements.qaResultsContainer.querySelector('.empty-qa-placeholder');
    if (placeholder) placeholder.remove();
    
    elements.qaResultsContainer.appendChild(userBubble);
    elements.qaResultsContainer.scrollTop = elements.qaResultsContainer.scrollHeight;
    
    // Loading indicator card
    const loadingCard = document.createElement('div');
    loadingCard.className = 'qa-bubble';
    loadingCard.innerHTML = `
        <div class="qa-response-title">Career Assistant</div>
        <div class="qa-passage-card" style="border-style:dashed;">
            <div class="passage-text" style="display:flex;align-items:center;gap:0.5rem">
                <i class="spinner" style="width:14px;height:14px;border-width:2px;margin:0"></i>
                <span>Retrieving semantic passages from vector index...</span>
            </div>
        </div>
    `;
    elements.qaResultsContainer.appendChild(loadingCard);
    elements.qaResultsContainer.scrollTop = elements.qaResultsContainer.scrollHeight;
    
    // Run retrieval
    try {
        const queryEmbedding = await getEmbedding(query);
        
        // Calculate similarity scores
        const results = chunksDB.map(chunk => {
            const similarity = cosineSimilarity(queryEmbedding, chunk.embedding);
            return {
                text: chunk.text,
                source: chunk.source,
                score: similarity
            };
        });
        
        // Sort descending and get top 3 matching chunks (matches faiss IndexFlatL2 search parameter)
        results.sort((a, b) => b.score - a.score);
        const top3 = results.slice(0, 3);
        
        // Remove loading spinner
        loadingCard.remove();
        
        // Render passages
        const responseGroup = document.createElement('div');
        responseGroup.className = 'qa-bubble';
        
        let responseHtml = `<div class="qa-response-title">Career Assistant</div>`;
        
        top3.forEach(match => {
            const matchScorePercent = Math.round((match.score + 1) * 50); // normalize [-1, 1] to [0, 100]
            responseHtml += `
                <div class="qa-passage-card">
                    <div class="passage-header">
                        <span class="passage-source">${match.source} Match</span>
                        <span class="passage-score">${matchScorePercent}% Relevance</span>
                    </div>
                    <p class="passage-text">"${match.text}"</p>
                </div>
            `;
        });
        
        responseGroup.innerHTML = responseHtml;
        elements.qaResultsContainer.appendChild(responseGroup);
        elements.qaResultsContainer.scrollTop = elements.qaResultsContainer.scrollHeight;
        
    } catch (error) {
        console.error('RAG Retrieval error:', error);
        loadingCard.remove();
        alert('Error retrieving search context.');
    }
}

// Cosine Similarity calculation function
function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let mA = 0;
    let mB = 0;
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        mA += a[i] * a[i];
        mB += b[i] * b[i];
    }
    return dotProduct / (Math.sqrt(mA) * Math.sqrt(mB));
}

// 9. PDF Report Generator (Matches PDF output fields in Python Notebook)
function generateAssessmentReport() {
    if (!resumeRawText || chunksDB.length === 0) return;
    
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    const today = new Date().toLocaleString();
    
    // Parse current assessed values
    const jdSkills = allKnownSkills.filter(skill => jdRawText.toLowerCase().includes(skill));
    const resumeSkills = allKnownSkills.filter(skill => resumeRawText.toLowerCase().includes(skill));
    const foundSkills = jdSkills.filter(skill => resumeSkills.includes(skill));
    const missingSkills = jdSkills.filter(skill => !resumeSkills.includes(skill));
    
    const matchScoreVal = jdSkills.length > 0 
        ? ((jdSkills.length - missingSkills.length) / jdSkills.length) * 100 
        : 100;
        
    const bestRoleVal = elements.lblRecRole.textContent;
    
    // Header block
    doc.setFillColor(6, 9, 19);
    doc.rect(0, 0, 210, 40, 'F');
    
    doc.setTextColor(255, 255, 255);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(22);
    doc.text('CareerRAG Smart HR Analytics', 14, 25);
    
    doc.setTextColor(148, 163, 184);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text('SEMANTIC RESUME ASSESSMENT & CAREER ASSISTANT REPORT', 14, 32);
    
    // Telemetry Info block
    doc.setTextColor(50, 50, 50);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.text('Assessment Telemetry Summary', 14, 55);
    
    doc.setDrawColor(226, 232, 240);
    doc.line(14, 58, 196, 58);
    
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(11);
    doc.setTextColor(80, 80, 80);
    
    doc.text(`Generated On: ${today}`, 14, 68);
    doc.text(`Candidate File: ${resumeFileName}`, 14, 76);
    doc.text(`Resume Match Score: ${matchScoreVal.toFixed(2)}%`, 14, 84);
    doc.text(`Recommended Career Role: ${bestRoleVal}`, 14, 92);
    
    // Skills Table
    doc.setTextColor(50, 50, 50);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.text('Skill Alignment Matrix', 14, 110);
    doc.line(14, 113, 196, 113);
    
    doc.setFontSize(11);
    doc.setFillColor(241, 245, 249);
    doc.rect(14, 120, 182, 8, 'F');
    doc.text('Skill Status', 20, 126);
    doc.text('Required Skills List', 100, 126);
    
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(80, 80, 80);
    
    // Row 1: Found
    doc.text('Found Skills', 20, 136);
    const foundStr = foundSkills.length > 0 ? foundSkills.join(', ').toUpperCase() : 'None';
    const splitFound = doc.splitTextToSize(foundStr, 100);
    doc.text(splitFound, 100, 136);
    
    let y = 136 + (splitFound.length * 6) + 4;
    
    // Row 2: Missing
    doc.setFillColor(248, 250, 252);
    doc.rect(14, y - 6, 182, 8 + (missingSkills.length > 0 ? 4 : 0), 'F');
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(80, 80, 80);
    doc.text('Missing Skills', 20, y);
    const missingStr = missingSkills.length > 0 ? missingSkills.join(', ').toUpperCase() : 'None';
    const splitMissing = doc.splitTextToSize(missingStr, 100);
    doc.text(splitMissing, 100, y);
    
    y = y + (splitMissing.length * 6) + 12;
    
    // Upskilling section
    doc.setTextColor(50, 50, 50);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.text('Upskilling Monthly Action Roadmap', 14, y);
    doc.line(14, y + 3, 196, y + 3);
    
    y += 12;
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.setTextColor(80, 80, 80);
    
    if (missingSkills.length === 0) {
        doc.text('• Candidate fulfills all prerequisites. Ready for evaluation.', 20, y);
        y += 8;
    } else {
        missingSkills.forEach((skill, i) => {
            let timelineRec = '';
            if (skill === 'sql') {
                timelineRec = 'Master SQL join syntax, window functions, and query optimization.';
            } else if (skill === 'tensorflow' || skill === 'pytorch') {
                timelineRec = `Build neural networks and custom models in ${skill}. Check tutorials.`;
            } else if (skill === 'docker') {
                timelineRec = 'Understand containerization, write Dockerfiles, and compose setups.';
            } else if (skill === 'aws') {
                timelineRec = 'Study cloud hosting, EC2 instances, S3 permissions, and IAM policies.';
            } else if (skill === 'machine learning' || skill === 'scikit-learn') {
                timelineRec = 'Study algorithms, regressions, ensemble estimators, and hyperparameter tuning.';
            } else {
                timelineRec = `Complete comprehensive training modules for ${skill.toUpperCase()}.`;
            }
            
            const roadmapText = `Month ${i+1} - Learn ${skill.toUpperCase()}: ${timelineRec}`;
            const splitRoadmapText = doc.splitTextToSize(roadmapText, 170);
            doc.text(splitRoadmapText, 20, y);
            y += (splitRoadmapText.length * 6) + 2;
        });
    }
    
    // Page footer
    doc.setFontSize(9);
    doc.setFont('helvetica', 'italic');
    doc.setTextColor(150, 150, 150);
    doc.text('Report compiled automatically by CareerRAG Vector Match Engine.', 14, 280);
    
    // Save
    doc.save('Career_Assessment_Report.pdf');
}

// 10. Helper UI Functions
function resetSystem() {
    resumeRawText = '';
    jdRawText = '';
    resumeFileName = '';
    chunksDB = [];
    isAnalyzing = false;
    
    elements.fileInput.value = '';
    elements.fileStatusBox.classList.add('hidden');
    elements.textPreviewContainer.style.display = 'none';
    elements.extractedTextPreview.innerHTML = '';
    
    elements.resumeUploadTitle.textContent = "Drag & drop resume PDF";
    elements.resumeUploadDesc.textContent = "or click to browse local files";
    elements.uploadZone.style.borderColor = "rgba(255, 255, 255, 0.08)";
    
    loadJdTemplate();
    
    // Reset Telemetry UI
    animateScoreGauge(0);
    elements.lblRecRole.textContent = 'Awaiting Analysis';
    elements.lblSkillsFoundCount.textContent = '0';
    elements.lblSkillsMissingCount.textContent = '0';
    
    elements.containerSkillsFound.innerHTML = '<span class="empty-placeholder">None detected</span>';
    elements.containerSkillsMissing.innerHTML = '<span class="empty-placeholder">None detected</span>';
    
    // Reset fit progress bars
    elements.roleFitBars.innerHTML = `
        <div class="role-item">
            <div class="role-meta"><span>Machine Learning Engineer</span><span class="role-pct">0%</span></div>
            <div class="progress-track"><div class="progress-bar" style="width: 0%"></div></div>
        </div>
        <div class="role-item">
            <div class="role-meta"><span>Data Scientist</span><span class="role-pct">0%</span></div>
            <div class="progress-track"><div class="progress-bar" style="width: 0%"></div></div>
        </div>
        <div class="role-item">
            <div class="role-meta"><span>AI Engineer</span><span class="role-pct">0%</span></div>
            <div class="progress-track"><div class="progress-bar" style="width: 0%"></div></div>
        </div>
        <div class="role-item">
            <div class="role-meta"><span>Backend Developer</span><span class="role-pct">0%</span></div>
            <div class="progress-track"><div class="progress-bar" style="width: 0%"></div></div>
        </div>
    `;
    
    // Reset learning roadmaps
    elements.roadmapTimelineContainer.innerHTML = `
        <div class="empty-timeline-placeholder">
            <i data-lucide="sparkles"></i>
            <p>Upskilling timelines will generate dynamically after assessment is complete.</p>
        </div>
    `;
    
    // Reset RAG console
    elements.lblVectorCount.textContent = '0 Vectors';
    elements.inputQuery.value = '';
    elements.inputQuery.setAttribute('disabled', 'true');
    elements.btnQuery.setAttribute('disabled', 'true');
    elements.qaResultsContainer.innerHTML = `
        <div class="empty-qa-placeholder">
            <i data-lucide="message-square"></i>
            <p>Search query matching passages across Resume and JD chunk embeddings.</p>
        </div>
    `;
    
    elements.btnDownloadPdf.setAttribute('disabled', 'true');
    enableInputsIfReady();
    lucide.createIcons();
}
