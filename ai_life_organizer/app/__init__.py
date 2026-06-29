import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_mail import Mail
from apscheduler.schedulers.background import BackgroundScheduler
from config import config

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()
mail = Mail()
scheduler = BackgroundScheduler(daemon=True)


def create_app(config_name="default"):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Init extensions
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    mail.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "warning"

    # Register blueprints
    from app.auth import auth_bp
    from app.dashboard import dashboard_bp
    from app.tasks import tasks_bp
    from app.goals import goals_bp
    from app.calendar import calendar_bp
    from app.notes import notes_bp
    from app.documents import documents_bp
    from app.ai_assistant import ai_bp
    from app.reminders import reminders_bp
    from app.analytics import analytics_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(goals_bp)
    app.register_blueprint(calendar_bp)
    app.register_blueprint(notes_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(ai_bp)
    app.register_blueprint(reminders_bp)
    app.register_blueprint(analytics_bp)

    # Start background scheduler
    from app.reminders.scheduler import trigger_due_reminders
    from app.goals.scheduler import mark_overdue_goals

    if not scheduler.running:
        scheduler.add_job(
            func=trigger_due_reminders,
            args=[app],
            trigger="interval",
            minutes=5,
            id="trigger_reminders",
        )
        scheduler.add_job(
            func=mark_overdue_goals,
            args=[app],
            trigger="cron",
            hour=0,
            minute=0,
            id="mark_overdue_goals",
        )
        scheduler.start()

    return app
