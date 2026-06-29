from datetime import datetime, date
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    tasks = db.relationship("Task", backref="owner", lazy=True, cascade="all, delete-orphan")
    goals = db.relationship("Goal", backref="owner", lazy=True, cascade="all, delete-orphan")
    events = db.relationship("CalendarEvent", backref="owner", lazy=True, cascade="all, delete-orphan")
    notes = db.relationship("Note", backref="owner", lazy=True, cascade="all, delete-orphan")
    documents = db.relationship("Document", backref="owner", lazy=True, cascade="all, delete-orphan")
    reminders = db.relationship("Reminder", backref="owner", lazy=True, cascade="all, delete-orphan")
    reset_tokens = db.relationship("PasswordResetToken", backref="owner", lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Task(db.Model):
    __tablename__ = "tasks"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    priority = db.Column(db.Enum("Low", "Medium", "High", "Critical"), default="Medium", nullable=False)
    status = db.Column(db.Enum("Pending", "In Progress", "Completed"), default="Pending", nullable=False)
    deadline = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Goal(db.Model):
    __tablename__ = "goals"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    target_date = db.Column(db.Date, nullable=False)
    progress = db.Column(db.Integer, default=0)  # 0-100
    status = db.Column(db.Enum("Active", "Achieved", "Overdue"), default="Active", nullable=False)
    achieved_at = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def update_progress(self, value):
        """Update progress and set status accordingly."""
        self.progress = max(0, min(100, int(value)))
        if self.progress == 100:
            self.status = "Achieved"
            self.achieved_at = date.today()
        elif self.target_date < date.today() and self.progress < 100:
            self.status = "Overdue"
        else:
            self.status = "Active"


class CalendarEvent(db.Model):
    __tablename__ = "calendar_events"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    start_datetime = db.Column(db.DateTime, nullable=False)
    end_datetime = db.Column(db.DateTime, nullable=True)
    event_type = db.Column(db.Enum("Event", "TaskDeadline", "GoalTarget"), default="Event", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Note(db.Model):
    __tablename__ = "notes"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text)
    category = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Document(db.Model):
    __tablename__ = "documents"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename = db.Column(db.String(255), nullable=False)       # stored filename (uuid-prefixed)
    original_name = db.Column(db.String(255), nullable=False)  # original upload name
    file_size = db.Column(db.Integer, nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    extracted_text = db.Column(db.Text)
    parse_failed = db.Column(db.Boolean, default=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)


class Reminder(db.Model):
    __tablename__ = "reminders"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    text = db.Column(db.String(500), nullable=False)
    remind_at = db.Column(db.DateTime, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    source_type = db.Column(db.Enum("Manual", "Task", "Goal", "Event"), default="Manual", nullable=False)
    source_id = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PasswordResetToken(db.Model):
    __tablename__ = "password_reset_tokens"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = db.Column(db.String(255), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
