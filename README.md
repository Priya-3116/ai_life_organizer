# 🤖 AI Life Organizer

An **AI-powered Personal Productivity & Life Management System** built with **Python, Flask, Machine Learning, and MySQL**. The application helps users organize their daily lives by managing tasks, tracking goals, building habits, logging moods, and receiving AI-driven productivity insights through a modern, responsive web dashboard.


## 📌 Overview

AI Life Organizer is a smart productivity platform that combines **Artificial Intelligence**, **Natural Language Processing (NLP)**, and **Machine Learning** to simplify personal planning and improve productivity.

The system enables users to:

* Manage daily tasks and recurring schedules
* Track personal and professional goals
* Build positive habits with streak tracking
* Record moods and analyze emotional trends
* Organize notes and documents
* Receive intelligent AI suggestions
* Visualize productivity through interactive dashboards



# ✨ Features

### 🔐 Authentication & Security

* Secure User Registration & Login
* Password Reset via Email
* Password Encryption
* CSRF Protection
* Input Sanitization

### 🤖 AI Features

* AI Assistant powered by Sentence Transformers
* Semantic Search
* Natural Language Task Creation
* Smart Note Linking
* Productivity Insights
* Mood Correlation Analysis

### ✅ Task Management

* Create, Edit & Delete Tasks
* Priority Levels
* Deadlines
* Recurring Tasks
* Subtasks
* Time Tracking
* Kanban Board

### 🎯 Goal Management

* Goal Creation
* Progress Tracking
* Automatic Status Updates
* Target Dates
* Goal Sharing

### 💪 Habit Tracker

* Daily Check-ins
* Streak Counter
* Habit History
* Progress Charts

### 😊 Mood Tracker

* Mood Logging
* Energy Level Tracking
* AI-Based Mood Analysis

### 📝 Notes & Documents

* Smart Notes
* Knowledge Graph Linking
* PDF Upload
* DOCX Upload
* Automatic Text Extraction

### 📅 Calendar

* Event Scheduling
* Task Deadlines
* Goal Milestones

### 📊 Analytics Dashboard

* Weekly Reviews
* Productivity Charts
* Habit Reports
* Mood Trends
* Goal Statistics

### 🍅 Focus Mode

* Pomodoro Timer
* Focus Sessions
* Interruption Tracking

### 📤 Data Management

* CSV Export
* JSON Export
* Data Import

### 📱 Progressive Web App

* Installable on Desktop
* Installable on Mobile
* Offline Support
* Dark & Light Theme

### 📧 Email Services

* Reminder Notifications
* Weekly Productivity Digest


# 🛠️ Tech Stack

 Category                Technology                          
 --------------------   ----------------------------------- 
 Programming Language   Python 3.10+                        
 Backend                Flask, Flask-SQLAlchemy             
 Machine Learning       Scikit-learn, Sentence Transformers 
 Database               MySQL, PyMySQL                      
 Frontend               HTML5, CSS3, JavaScript, Bootstrap  
 Charts                 Chart.js                            
 Scheduler              APScheduler                         
 Authentication         Flask-Login                         
 Security               Flask-WTF, Werkzeug, Bleach         
 Email                  Flask-Mail                          
 Document Parsing       PyPDF2, python-docx                 
 Scientific Libraries   NumPy, SciPy                        


# 📂 Project Structure

ai_life_organizer/
│
├── app.py
├── run.py
├── config.py
├── requirements.txt
├── v2_features.py
│
├── static/
│   ├── manifest.json
│   └── sw.js
│
├── uploads/
│
└── app/
    ├── auth/
    ├── dashboard/
    ├── tasks/
    ├── goals/
    ├── calendar/
    ├── notes/
    ├── documents/
    ├── ai_assistant/
    ├── reminders/
    ├── analytics/
    ├── models.py
    └── __init__.py


# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YourUsername/AI-Life-Organizer.git
cd AI-Life-Organizer
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Create MySQL Database

```sql
CREATE DATABASE ai_life_organizer;
```

## Configure Environment Variables

Create a `.env` file.

```env
SECRET_KEY=your-secret-key

DATABASE_URL=mysql+pymysql://root:password@localhost/ai_life_organizer

MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your@gmail.com
MAIL_PASSWORD=your-app-password
MAIL_DEFAULT_SENDER=your@gmail.com
```

## Run Application

python app.py


Visit:

http://127.0.0.1:5000

# 📷 Screenshots

* Login Page

  <img width="496" height="475" alt="Screenshot 2026-06-29 140343" src="https://github.com/user-attachments/assets/325f982f-cdc5-402d-9eae-2cb6714472a5" />

* Dashboard

  <img width="1219" height="622" alt="Screenshot 2026-06-29 140433" src="https://github.com/user-attachments/assets/78fcd640-6c08-4229-9bb7-dd3b2854d089" />

* AI Assistant

   <img width="829" height="373" alt="Screenshot 2026-06-29 140826" src="https://github.com/user-attachments/assets/34e30c68-3b2a-464b-be44-2c96ee5af205" />

* Task Manager

   <img width="1143" height="405" alt="Screenshot 2026-06-29 142201" src="https://github.com/user-attachments/assets/6f5e2bb2-4939-477e-b120-4f203001625c" />

* Habit Tracker

  <img width="606" height="405" alt="Screenshot 2026-06-29 141010" src="https://github.com/user-attachments/assets/157c6340-6b6e-4cfd-841a-66703b259ff4" />
  <img width="390" height="222" alt="Screenshot 2026-06-29 141043" src="https://github.com/user-attachments/assets/2b77c443-b241-4404-b972-465437ce9fce" />


* Analytics Dashboard

  <img width="1171" height="597" alt="Screenshot 2026-06-29 142527" src="https://github.com/user-attachments/assets/40068bb7-bcd2-4e14-b900-84195130bbbb" />

* Mood Insights

  <img width="1139" height="497" alt="Screenshot 2026-06-29 141321" src="https://github.com/user-attachments/assets/7e9b7d2f-562e-43ca-9dce-aeb2c995d570" />

* Kanban Board

  <img width="1154" height="458" alt="Screenshot 2026-06-29 141654" src="https://github.com/user-attachments/assets/7880bb50-b322-4d89-bae5-5fb84f8e4b8e" />



# 🗄️ Database Models

* User
* Task
* Goal
* CalendarEvent
* Reminder
* Habit
* HabitCheckin
* MoodLog
* MoodInsight
* WeeklyReview
* FocusSession
* Note
* NoteLink
* Document
* Project
* TaskShare
* GoalShare
* TimeLog
* Subtask


# ⏱️ Background Scheduler

The application automatically performs:

* Reminder email notifications every **5 minutes**
* Daily goal status updates
* Weekly productivity report generation
* Weekly email digest on **Sunday**


# 📱 Progressive Web App

The application supports:

* Install on Windows
* Install on Android
* Install on macOS
* Install on iOS
* Offline Access
* Fast Loading


# 🚀 Future Enhancements

* Voice Assistant
* AI Chatbot
* OCR Image Text Extraction
* Google Calendar Integration
* Mobile Application
* Team Collaboration Workspace
* AI Productivity Score
* Personalized Productivity Recommendations


# 👩‍💻 Team

Priya Hulamani


# 📄 License

This project is developed for **educational and academic purposes** only.


# ⭐ Support

If you found this project helpful, please consider **starring** the repository.


