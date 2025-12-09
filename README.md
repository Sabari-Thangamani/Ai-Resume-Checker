AI Resume Checker
AI Resume Checker is a Flask-based web application that helps users analyse and improve their resumes for specific job roles. It extracts text from PDF/DOCX files, compares the resume against predefined or custom job descriptions, and generates a match score, missing keywords, ATS tips, and improvement suggestions.​

Features
User authentication with registration, login, profile, and optional premium flag.​

Resume upload (PDF/DOCX), text extraction, TF‑IDF based similarity score, missing skill detection, and ATS checks (sections, keyword density, tips).​

AI assistance using spaCy, transformers, and optional OpenAI API for skill extraction, resume summary, section rewriting, cover letter generation, and proofreading.​

PDF report export with score, missing skills, suggestions, and basic skill comparison charts.​

Built‑in quizzes for roles like Data Analyst and Software Engineer, with score tracking, weak‑area detection, and stored quiz history per user.​

Tech Stack
Backend: Flask, SQLAlchemy, Flask‑Login, Flask‑WTF.​

NLP & AI: spaCy, HuggingFace transformers, scikit‑learn, optional OpenAI API.​

Utilities: PyPDF2, python‑docx, ReportLab, Matplotlib.​

Database: Relational DB (SQLite/PostgreSQL compatible models for users, resumes, analyses, custom jobs, quizzes, phrases, analytics).​

Setup (Basic)
Create and activate a virtual environment, then install dependencies: pip install -r requirements.txt.​

Configure environment variables in .env (database URL, secret key, optional OpenAI API key) and run the app with python run_app.py or directly via app.py after initializing the database.
