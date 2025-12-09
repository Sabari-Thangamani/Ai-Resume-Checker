# app.py - Main Flask application for AI Resume Checker
# This handles routing, file upload, text extraction, similarity calculation, and rendering results.

import io
import os
import random
import json
import re
import logging
from flask import Flask, render_template, request, flash, redirect, url_for, session, send_file, jsonify, g
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from bs4 import BeautifulSoup
import requests
from langdetect import detect
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from openai import OpenAI  # type: ignore
from transformers import pipeline  # type: ignore
from reportlab.pdfgen import canvas  # type: ignore
from reportlab.lib.pagesizes import letter  # type: ignore
from reportlab.lib.utils import ImageReader  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from config import Config
from models import db, User, Resume, CustomJob, Analysis, QuizResult, Phrase, Analytics
from forms import RegisterForm, LoginForm, UploadForm, ProfileForm, LinkedInForm, RewriteForm, CoverForm

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
csrf = CSRFProtect(app)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load spaCy model (optional)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Warning: Could not load spaCy model: {e}")
    nlp = None

# Initialize OpenAI client (set API key in environment variable OPENAI_API_KEY)
openai_client = None

# Initialize HuggingFace pipeline for skill extraction
try:
    skill_extractor = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
except Exception as e:
    print(f"Warning: Could not load HuggingFace pipeline: {e}")
    skill_extractor = None

# Folder for job descriptions
JOBS_FOLDER = 'jobs'



def extract_text_from_pdf(file_content):
    """Extract text from PDF using PyPDF2."""
    pdf_reader = PdfReader(io.BytesIO(file_content))
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() + '\n'
    return text.lower().strip()

def extract_text_from_docx(file_content):
    """Extract text from DOCX using python-docx."""
    doc = Document(io.BytesIO(file_content))
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text.lower().strip()

def calculate_similarity(resume_text, job_desc):
    """Calculate TF-IDF cosine similarity between resume and job description as percentage."""
    if not resume_text or not job_desc:
        return 0.0
    documents = [resume_text, job_desc]
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)  # Return as percentage

def get_missing_skills(resume_text, job_desc):
    """Extract simple keywords from job description and find missing ones in resume."""
    resume_words = set(resume_text.lower().split())
    # Extract keywords: words >2 chars, alphabetic only
    job_keywords = set(word for word in job_desc.lower().split() if len(word) > 2 and word.isalpha())
    missing = list(job_keywords - resume_words)
    # Limit to top 10 for display
    return sorted(missing[:10])

def extract_skills(text):
    """Extract top skills from text using HuggingFace transformers or spaCy fallback."""
    if skill_extractor:
        try:
            entities = skill_extractor(text)
            skills = [entity['word'] for entity in entities if entity['entity_group'] == 'MISC' or entity['entity_group'] == 'ORG']  # Adjust for skills
        except Exception as e:
            skills = []
    else:
        skills = []
    # Fallback to spaCy if needed
    if not skills:
        doc = nlp(text)
        skills = [token.lemma_ for token in doc if token.pos_ == "NOUN" and len(token.lemma_) > 2]
    return list(set(skills))[:10]

def generate_summary(resume_text, job_desc):
    """Generate a short resume summary using OpenAI."""
    prompt = f"Generate a concise professional summary for a resume based on the following text: {resume_text[:500]}. Tailor it for a job requiring: {job_desc[:200]}. Make it unique and creative. Random variation: {random.randint(1,1000)}"
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to spaCy with randomization
            doc = nlp(resume_text)
            nouns = [token.text for token in doc if token.pos_ == "NOUN" and len(token.text) > 3]
            random.shuffle(nouns)
            nouns = nouns[:5]
            summary = f"A skilled professional with expertise in {', '.join(nouns)}. Well-suited for roles requiring {job_desc[:50]}..."
    else:
        # Fallback to spaCy with randomization
        doc = nlp(resume_text)
        nouns = [token.text for token in doc if token.pos_ == "NOUN" and len(token.text) > 3]
        random.shuffle(nouns)
        nouns = nouns[:5]
        summary = f"A skilled professional with expertise in {', '.join(nouns)}. Well-suited for roles requiring {job_desc[:50]}..."
    return summary

def generate_pdf(score, missing_skills, suggestion, job_role, resume_skills=None, job_skills=None):
    """Generate PDF report using ReportLab, with optional chart."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, f"Resume Analysis Report for {job_role}")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 80, f"Match Score: {score}%")
    c.drawString(100, height - 100, f"Missing Skills: {', '.join(missing_skills)}")
    c.drawString(100, height - 120, f"Suggestion: {suggestion}")

    # Add simple chart if skills provided
    if resume_skills and job_skills:
        fig, ax = plt.subplots()
        ax.bar(resume_skills[:5], [1]*len(resume_skills[:5]), label='Resume Skills')
        ax.bar(job_skills[:5], [0.5]*len(job_skills[:5]), label='Job Skills')
        ax.legend()
        chart_buffer = io.BytesIO()
        fig.savefig(chart_buffer, format='png')
        chart_buffer.seek(0)
        c.drawImage(ImageReader(chart_buffer), 100, height - 300, width=300, height=200)
        plt.close(fig)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def ats_checks(resume_text, job_desc):
    """Perform ATS-specific checks: keyword density, section detection."""
    job_words = set(job_desc.lower().split())
    resume_words = set(resume_text.lower().split())
    density = len(job_words & resume_words) / len(job_words) * 100 if job_words else 0
    ats_score = round(density, 2)
    sections = ['experience', 'education', 'skills', 'summary']
    detected = [s for s in sections if re.search(r'\b' + s + r'\b', resume_text, re.I)]
    tips = []
    if not detected:
        tips.append("Add standard sections like Experience, Education, Skills.")
    if ats_score < 50:
        tips.append("Increase keyword density by matching job description terms.")
    return ats_score, tips

def rewrite_resume_section(section_text, job_desc):
    """Use OpenAI to rewrite a resume section."""
    prompt = f"Rewrite this resume section to better match the job description: {job_desc[:200]}. Original: {section_text}"
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return section_text
    return section_text

def generate_cover_letter(resume_text, job_desc):
    """Generate cover letter using OpenAI."""
    prompt = f"Generate a professional cover letter based on resume: {resume_text[:300]} for job: {job_desc[:200]}"
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "Cover letter generation failed."
    return "Cover letter generation requires OpenAI API key."

def import_linkedin(url):
    """Import data from LinkedIn public profile."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        name = soup.find('h1').text.strip() if soup.find('h1') else 'Unknown'
        experience = soup.find('section', {'class': 'experience'}).text.strip() if soup.find('section', {'class': 'experience'}) else ''
        skills = soup.find('section', {'class': 'skills'}).text.strip() if soup.find('section', {'class': 'skills'}) else ''
        return {'name': name, 'experience': experience, 'skills': skills}
    except Exception as e:
        return None

def proofread_text(text):
    """Proofread text using OpenAI."""
    prompt = f"Proofread and suggest improvements for this text: {text[:500]}"
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "Proofreading requires OpenAI API key."
    return "Proofreading requires OpenAI API key."

def get_phrases(section):
    """Get pre-written phrases for a section."""
    phrases = Phrase.query.filter_by(section=section).all()
    return [p.content for p in phrases]

def enhance_score(match_score, ats_score, completeness):
    """Combine scores for overall."""
    return round((match_score * 0.4 + ats_score * 0.3 + completeness * 0.3), 2)

@app.route('/')
def index():
    """Render the homepage with upload form."""
    form = UploadForm(meta={'csrf': False})
    return render_template('index.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration."""
    if current_user.is_authenticated:
        logging.info(f"Authenticated user {current_user.id} attempted to access register page.")
        return redirect(url_for('dashboard'))
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(email=form.email.data, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        logging.info(f"New user registered: {form.email.data}")
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if current_user.is_authenticated:
        logging.info(f"Authenticated user {current_user.id} attempted to access login page.")
        return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            logging.info(f"User {user.id} logged in successfully.")
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            logging.warning(f"Failed login attempt for email: {form.email.data}")
            flash('Invalid email or password.')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """User logout."""
    logging.info(f"User {current_user.id} logged out.")
    logout_user()
    flash('Logged out successfully.')
    return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile management."""
    form = ProfileForm()
    if form.validate_on_submit():
        current_user.is_premium = form.is_premium.data
        db.session.commit()
        logging.info(f"User {current_user.id} updated profile.")
        flash('Profile updated.')
    form.is_premium.data = current_user.is_premium
    return render_template('profile.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard."""
    resumes = Resume.query.filter_by(user_id=current_user.id).all()
    analyses = Analysis.query.filter_by(user_id=current_user.id).all()
    quiz_results = QuizResult.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', resumes=resumes, analyses=analyses, quiz_results=quiz_results)

@app.route('/upload_custom_job', methods=['POST'])
@login_required
def upload_custom_job():
    """Upload custom job description."""
    title = request.form.get('title')
    description = request.form.get('description')
    job = CustomJob(user_id=current_user.id, title=title, description=description)
    db.session.add(job)
    db.session.commit()
    logging.info(f"User {current_user.id} added custom job: {title}")
    flash('Custom job added.')
    return redirect(url_for('dashboard'))

@app.route('/linkedin_import', methods=['GET', 'POST'])
@login_required
def linkedin_import():
    """Import from LinkedIn."""
    form = LinkedInForm()
    if form.validate_on_submit():
        data = import_linkedin(form.url.data)
        if data:
            resume = Resume(user_id=current_user.id, filename='LinkedIn Import', text_extracted=f"{data['name']}\n{data['experience']}\n{data['skills']}")
            db.session.add(resume)
            db.session.commit()
            logging.info(f"User {current_user.id} imported resume from LinkedIn.")
            flash('Imported from LinkedIn.')
            return redirect(url_for('dashboard'))
        flash('Import failed.')
    return render_template('linkedin_import.html', form=form)

@app.route('/rewrite_resume', methods=['POST'])
@login_required
def rewrite_resume():
    """Rewrite resume section."""
    section = request.form.get('section')
    job_desc = request.form.get('job_desc')
    rewritten = rewrite_resume_section(section, job_desc)
    logging.info(f"User {current_user.id} rewrote resume section.")
    return jsonify({'rewritten': rewritten})

@app.route('/generate_cover', methods=['POST'])
@login_required
def generate_cover():
    """Generate cover letter."""
    form = CoverForm()
    if form.validate_on_submit():
        resume = Resume.query.get(form.resume_id.data)
        job = CustomJob.query.get(form.job_id.data)
        if resume and job:
            cover = generate_cover_letter(resume.text_extracted, job.description)
            logging.info(f"User {current_user.id} generated cover letter for resume {resume.id} and job {job.id}.")
            return render_template('cover_letter.html', cover=cover)
    flash('Generation failed.')
    return redirect(url_for('dashboard'))

@app.route('/proofread', methods=['POST'])
@login_required
def proofread():
    """Proofread text."""
    text = request.form.get('text')
    suggestions = proofread_text(text)
    logging.info(f"User {current_user.id} proofread text.")
    return jsonify({'suggestions': suggestions})

@app.route('/export_improved/<int:resume_id>')
@login_required
def export_improved(resume_id):
    """Export improved resume."""
    resume = Resume.query.get(resume_id)
    if not resume or resume.user_id != current_user.id:
        flash('Not found.')
        return redirect(url_for('dashboard'))
    pdf_content = generate_pdf(100, [], 'Improved resume', 'Improved', [], [])
    return send_file(io.BytesIO(pdf_content), as_attachment=True, download_name='improved_resume.pdf', mimetype='application/pdf')

@app.route('/interview_prep/<job_role>')
@login_required
def interview_prep(job_role):
    """Interview preparation."""
    # Placeholder for more roles
    return redirect(url_for('quiz', job_role=job_role))


@app.route('/analyze', methods=['POST'])
def analyze_resume():
    """Handle resume upload, extract text, compare with job description, and render results."""
    logging.info("Starting resume analysis.")
    form = UploadForm(meta={'csrf': False})
    if form.validate_on_submit():
        file = form.resume.data
        job_role = form.job_role.data
        logging.info(f"Form validated. Job role: {job_role}, File: {file.filename}")
    else:
        logging.warning("Form validation failed.")
        flash('Form validation failed.')
        return redirect(url_for('index'))

    if not file or file.filename == '':
        logging.warning("No file selected.")
        flash('No file selected.')
        return redirect(url_for('index'))

    if not job_role:
        logging.warning("No job role selected.")
        flash('Please select a job role.')
        return redirect(url_for('index'))

    # Load job description from file
    job_file_path = os.path.join(JOBS_FOLDER, f'{job_role}.txt')
    if not os.path.exists(job_file_path):
        logging.error(f"Job description file not found: {job_file_path}")
        flash('Job description not found.')
        return redirect(url_for('index'))

    with open(job_file_path, 'r', encoding='utf-8') as f:
        job_desc = f.read().strip()
    logging.info(f"Loaded job description for {job_role}.")

    # Read file content
    file_content = file.read()
    filename = file.filename.lower()

    # Extract text based on file type
    if filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(file_content)
        logging.info("Extracted text from PDF.")
    elif filename.endswith('.docx'):
        resume_text = extract_text_from_docx(file_content)
        logging.info("Extracted text from DOCX.")
    else:
        logging.warning(f"Unsupported file type: {filename}")
        flash('Unsupported file type. Please upload PDF or DOCX.')
        return redirect(url_for('index'))

    if not resume_text:
        logging.error("Could not extract text from the file.")
        flash('Could not extract text from the file.')
        return redirect(url_for('index'))

    # Calculate similarity and missing skills
    match_score = calculate_similarity(resume_text, job_desc)
    missing_skills = get_missing_skills(resume_text, job_desc)
    logging.info(f"Calculated match score: {match_score}%, Missing skills: {len(missing_skills)}")

    # AI features
    top_skills = extract_skills(resume_text)
    ai_insights = {"top_skills": top_skills, "missing": missing_skills}
    summary = generate_summary(resume_text, job_desc)
    logging.info("Generated AI insights and summary.")

    # Generate suggestion
    if missing_skills:
        suggestion = f"Add more keywords related to {', '.join(missing_skills[:5])}. This will improve your match score!"
    else:
        suggestion = "Your resume is well-matched to the role! Consider tailoring it further for specific experiences."

    # Store for PDF and charts
    session['analysis'] = {'score': match_score, 'missing_skills': missing_skills, 'suggestion': suggestion, 'job_role': job_role.replace('_', ' ').title()}
    session['resume_skills'] = top_skills
    session['job_skills'] = extract_skills(job_desc)  # Job skills for comparison
    session['resume_text'] = resume_text
    session['job_desc'] = job_desc

    # Save to DB if logged in
    if current_user.is_authenticated:
        resume = Resume(user_id=current_user.id, filename=file.filename, text_extracted=resume_text)
        db.session.add(resume)
        db.session.commit()
        analysis = Analysis(user_id=current_user.id, resume_id=resume.id, job_role=job_role, score=match_score, missing_skills=json.dumps(missing_skills), suggestion=suggestion)
        db.session.add(analysis)
        db.session.commit()
        logging.info(f"Saved analysis to DB for user {current_user.id}.")

    logging.info("Resume analysis completed successfully.")
    # Render results page
    return render_template('results.html',
                           score=match_score,
                           missing_skills=missing_skills,
                           suggestion=suggestion,
                           job_role=job_role.replace('_', ' ').title(),
                           top_skills=top_skills,
                           ai_insights=ai_insights,
                           summary=summary)

# Full question pools
DATA_ANALYST_QUESTIONS = [
    {"question": "What is the primary purpose of data cleaning?", "options": ["To make data look pretty", "To remove errors and inconsistencies", "To add more data", "To encrypt data"], "answer": 1},
    {"question": "Which tool is commonly used for data visualization?", "options": ["Excel", "Tableau", "Word", "PowerPoint"], "answer": 1},
    {"question": "What does SQL stand for?", "options": ["Simple Query Language", "Structured Query Language", "System Query Language", "Standard Query Language"], "answer": 1},
    {"question": "What is a primary key in a database?", "options": ["A key that opens the database", "A unique identifier for a record", "A key for encryption", "A backup key"], "answer": 1},
    {"question": "Which Python library is used for data analysis?", "options": ["NumPy", "Pandas", "Matplotlib", "Scikit-learn"], "answer": 1},
    {"question": "What does ETL stand for?", "options": ["Extract, Transform, Load", "Edit, Test, Launch", "Error, Test, Log", "Execute, Transfer, Link"], "answer": 0},
    {"question": "Which chart is best for showing trends over time?", "options": ["Pie chart", "Bar chart", "Line chart", "Scatter plot"], "answer": 2},
    {"question": "What is data normalization?", "options": ["Making data normal", "Scaling data to a standard range", "Normalizing text", "Creating normal distributions"], "answer": 1},
    {"question": "Which SQL command is used to retrieve data?", "options": ["INSERT", "UPDATE", "SELECT", "DELETE"], "answer": 2},
    {"question": "What is a data warehouse?", "options": ["A place to store data", "A large repository for structured data", "A cloud storage", "A database for images"], "answer": 1},
    {"question": "What is the difference between supervised and unsupervised learning?", "options": ["Supervised uses labeled data", "Unsupervised uses labeled data", "Both use labeled data", "Neither uses data"], "answer": 0},
    {"question": "Which statistical measure indicates the spread of data?", "options": ["Mean", "Median", "Standard Deviation", "Mode"], "answer": 2},
    {"question": "What is a JOIN in SQL?", "options": ["A way to combine tables", "A sorting method", "A filtering tool", "A calculation function"], "answer": 0},
    {"question": "Which library is used for machine learning in Python?", "options": ["Pandas", "NumPy", "Scikit-learn", "Matplotlib"], "answer": 2},
    {"question": "What does KPI stand for?", "options": ["Key Performance Indicator", "Known Process Improvement", "Key Project Initiative", "Known Performance Index"], "answer": 0},
    {"question": "Which data type is used for categorical variables?", "options": ["Integer", "Float", "String", "Boolean"], "answer": 2},
    {"question": "What is overfitting in machine learning?", "options": ["Model performs well on training data but poorly on new data", "Model performs poorly on training data", "Model is too simple", "Model has no errors"], "answer": 0},
    {"question": "Which tool is used for big data processing?", "options": ["Excel", "Hadoop", "Word", "PowerPoint"], "answer": 1},
    {"question": "What is a histogram used for?", "options": ["Showing relationships", "Showing distributions", "Showing trends", "Showing categories"], "answer": 1},
    {"question": "What is the purpose of A/B testing?", "options": ["To compare two versions", "To clean data", "To visualize data", "To store data"], "answer": 0}
]

SOFTWARE_ENGINEER_QUESTIONS = [
    {"question": "What is the main purpose of version control?", "options": ["To track changes in code", "To compile code", "To debug code", "To deploy code"], "answer": 0},
    {"question": "Which language is primarily used for web development?", "options": ["Python", "JavaScript", "C++", "Java"], "answer": 1},
    {"question": "What does API stand for?", "options": ["Application Programming Interface", "Advanced Programming Interface", "Automated Programming Interface", "Application Process Interface"], "answer": 0},
    {"question": "What is object-oriented programming?", "options": ["Programming with objects", "A paradigm using classes and objects", "Programming for objects", "Object-based coding"], "answer": 1},
    {"question": "Which HTTP method is used to retrieve data?", "options": ["POST", "PUT", "GET", "DELETE"], "answer": 2},
    {"question": "What is a bug in software?", "options": ["A feature", "An error or flaw", "A test case", "A comment"], "answer": 1},
    {"question": "Which tool is used for debugging?", "options": ["Compiler", "Debugger", "Linker", "Assembler"], "answer": 1},
    {"question": "What does MVC stand for?", "options": ["Model View Controller", "Main View Component", "Module View Code", "Method Variable Class"], "answer": 0},
    {"question": "Which language is statically typed?", "options": ["Python", "JavaScript", "Java", "Ruby"], "answer": 2},
    {"question": "What is continuous integration?", "options": ["Integrating code continuously", "A process of merging code changes frequently", "Continuous coding", "Integration testing"], "answer": 1},
    {"question": "What is the purpose of a constructor in OOP?", "options": ["To destroy objects", "To initialize objects", "To compare objects", "To print objects"], "answer": 1},
    {"question": "Which data structure is LIFO?", "options": ["Queue", "Stack", "Array", "List"], "answer": 1},
    {"question": "What does IDE stand for?", "options": ["Integrated Development Environment", "Interactive Design Editor", "Internal Data Encoder", "Independent Debugging Engine"], "answer": 0},
    {"question": "Which protocol is used for secure web communication?", "options": ["HTTP", "FTP", "HTTPS", "SMTP"], "answer": 2},
    {"question": "What is recursion in programming?", "options": ["A loop", "A function calling itself", "A variable", "A class"], "answer": 1},
    {"question": "Which algorithm is used for sorting?", "options": ["Bubble Sort", "Linear Search", "Binary Tree", "Hash Table"], "answer": 0},
    {"question": "What is the purpose of unit testing?", "options": ["To test the entire application", "To test individual components", "To deploy code", "To write code"], "answer": 1},
    {"question": "Which language is interpreted?", "options": ["C++", "Java", "Python", "C"], "answer": 2},
    {"question": "What is polymorphism in OOP?", "options": ["Multiple forms", "Single form", "No forms", "Formless"], "answer": 0},
    {"question": "Which tool is used for version control?", "options": ["Git", "Docker", "Jenkins", "Kubernetes"], "answer": 0}
]

@app.route('/quiz/<job_role>')
def quiz(job_role):
    """Render the quiz page for the selected job role."""
    if job_role == 'data_analyst':
        full_questions = DATA_ANALYST_QUESTIONS
    elif job_role == 'software_engineer':
        full_questions = SOFTWARE_ENGINEER_QUESTIONS
    else:
        full_questions = []

    questions = random.sample(full_questions, 10) if len(full_questions) >= 10 else full_questions
    session['questions'] = questions
    session['job_role'] = job_role

    return render_template('quiz.html', job_role=job_role.replace('_', ' ').title(), questions=questions)

@app.route('/quiz_submit', methods=['POST'])
def quiz_submit():
    """Process quiz submission and render results."""
    questions = session.get('questions', [])
    job_role = session.get('job_role', '')
    correct_count = 0
    wrong_count = 0
    user_answers = []
    for i in range(len(questions)):
        answer = request.form.get(f'q{i+1}')
        if answer is None:
            wrong_count += 1
            user_answers.append(None)
        else:
            user_answers.append(int(answer))
            if int(answer) == questions[i]['answer']:
                correct_count += 1
            else:
                wrong_count += 1
    show_retest = wrong_count <= 5

    # Track quiz scores for trends
    if 'quiz_scores' not in session:
        session['quiz_scores'] = []
    session['quiz_scores'].append(correct_count)
    if len(session['quiz_scores']) > 5:  # Keep last 5 scores
        session['quiz_scores'] = session['quiz_scores'][-5:]

    # Skill recommendations based on wrong answers
    wrong_indices = [i for i, ans in enumerate(user_answers) if ans != questions[i]['answer'] and ans is not None]
    weak_areas = [questions[i]['question'][:50] + '...' for i in wrong_indices]
    recommendations = f"Review these areas: {', '.join(weak_areas[:3])}" if weak_areas else "Great job! No weak areas identified."

    # Save to DB if logged in
    if current_user.is_authenticated:
        quiz_result = QuizResult(user_id=current_user.id, job_role=job_role, score=correct_count, total_questions=len(questions), recommendations=recommendations)
        db.session.add(quiz_result)
        db.session.commit()

    return render_template('quiz_results.html', correct=correct_count, wrong=wrong_count, questions=questions, user_answers=user_answers, job_role=job_role, show_retest=show_retest, quiz_scores=session['quiz_scores'], recommendations=recommendations)

@app.route('/quiz/<job_role>/retest')
def quiz_retest(job_role):
    """Redirect to a new quiz for retest."""
    return redirect(url_for('quiz', job_role=job_role))

@app.route('/chart_data/<chart_type>')
def chart_data(chart_type):
    """Return JSON data for charts."""
    if chart_type == 'skills':
        # Radar chart data for resume vs job skills
        resume_skills = session.get('resume_skills', [])
        job_skills = session.get('job_skills', [])
        data = {
            'labels': resume_skills[:5] + job_skills[:5],
            'datasets': [
                {'label': 'Resume', 'data': [1]*len(resume_skills[:5]) + [0]*len(job_skills[:5])},
                {'label': 'Job', 'data': [0]*len(resume_skills[:5]) + [1]*len(job_skills[:5])}
            ]
        }
    elif chart_type == 'quiz_trends':
        scores = session.get('quiz_scores', [])
        data = {
            'labels': [f'Quiz {i+1}' for i in range(len(scores))],
            'datasets': [{'label': 'Scores', 'data': scores}]
        }
    else:
        data = {}
    return jsonify(data)

@app.route('/generate_summary_ajax')
def generate_summary_ajax():
    """AJAX endpoint to generate AI summary."""
    resume_text = session.get('resume_text', '')
    job_desc = session.get('job_desc', '')
    if not resume_text or not job_desc:
        return jsonify({'summary': 'No data available.'})
    summary = generate_summary(resume_text, job_desc)
    return jsonify({'summary': summary})

@app.route('/download_report')
def download_report():
    """Download the analysis report as PDF."""
    analysis = session.get('analysis', {})
    if not analysis:
        flash('No analysis data found.')
        return redirect(url_for('index'))

    resume_skills = session.get('resume_skills', [])
    job_skills = session.get('job_skills', [])
    pdf_content = generate_pdf(analysis['score'], analysis['missing_skills'], analysis['suggestion'], analysis['job_role'], resume_skills, job_skills)
    return send_file(io.BytesIO(pdf_content), as_attachment=True, download_name='resume_report.pdf', mimetype='application/pdf')

@app.route('/contact', methods=['POST'])
def contact():
    """Handle contact form submission."""
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')

    if not name or not email or not message:
        flash('All fields are required.')
        return redirect(url_for('index') + '#contact')

    # Here you could send an email, but for now, just flash a success message
    flash(f'Thank you for your message, {name}! We will get back to you soon at {email}.')
    return redirect(url_for('index') + '#contact')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Seed phrases if not exist
        if not Phrase.query.first():
            phrases = [
                Phrase(section='summary', content='Experienced professional with a proven track record in delivering high-quality results.'),
                Phrase(section='experience', content='Led a team of 5 developers to successfully launch a new product feature.'),
                Phrase(section='skills', content='Proficient in Python, JavaScript, and SQL with expertise in web development.'),
                Phrase(section='education', content='Bachelor of Science in Computer Science from XYZ University.'),
            ]
            db.session.add_all(phrases)
            db.session.commit()
    print("Starting Flask app on http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', port=5000, debug=True)
