# MockMate-an-AI-Interviwer
# **🎤 Mock Interview System with Resume Analysis**
## **🌟 Overview**
This is a Flask-based mock interview system that helps users practice interview questions, submit voice responses, and receive confidence scores. It also supports resume uploading and automatic text feature extraction from resumes.

## **✨ Features**
🔐 Login system with session management

📄 Resume upload and text extraction from PDF/DOCX files

🧠 Skill, project, certification, and experience detection from resumes

🎙️ Audio recording and confidence scoring for interview questions

💬 Emotion recognition pipeline 

🎛️ Randomized question fetching from PostgreSQL database

📊 Performance reporting and PDF report download

## **📁 Project Structure**

<pre> ## 📁 Project Structure mock-interview-system/ │ ├── app.py # Main Flask application ├── requirements.txt # Project dependencies ├── README.md # Project documentation │ ├── templates/ # HTML templates for UI │ ├── login.html │ ├── upload_resume.html │ ├── index.html │ └── report.html │ ├── static/ # Static assets like uploads & recordings │ ├── uploads/ # Uploaded resumes (PDF/DOCX) │ └── recordings/ # User audio responses </pre>

## 🚀 Technologies Used

| Technology     | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| Flask          | Web framework to build backend and serve frontend                       |
| PostgreSQL     | Relational database to store interview questions                        |
| Transformers   | NLP models for speech recognition and emotion detection                 |
|               | - `openai/whisper-tiny.en`: ASR (Automatic Speech Recognition)           |
|               | - `j-hartmann/emotion-english-distilroberta-base`: Emotion classification|
| Pytesseract    | OCR engine for image-based text extraction (optional)                   |
| PyPDF2         | Extract text from PDF resumes                                           |
| python-docx    | Extract text from DOCX resumes                                          |
| FPDF           | Generate downloadable PDF interview reports                             |
| HTML/CSS/JS    | Frontend for UI, recording, and interactions                            |



# **OUTPUTS**
![Screenshot 2025-04-14 213020](https://github.com/user-attachments/assets/832fe67e-90c6-451d-96d4-05893ed15efc)
![Screenshot 2025-04-14 213050](https://github.com/user-attachments/assets/507a91d4-765f-4f1d-8044-0a52efeb43f4)
![Screenshot 2025-04-14 213147](https://github.com/user-attachments/assets/62b6fb6e-d052-47e6-8bec-0e56c0fd20e9)
![Screenshot 2025-04-14 213155](https://github.com/user-attachments/assets/2bf7075c-dccb-4742-b81d-d73d5e26bc98)



