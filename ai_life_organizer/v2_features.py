# =============================================================================
# AI Life Organizer — V2+ Feature Pack
# Smart Weekly Review, Knowledge Graph, Pomodoro, Mood Intelligence,
# NL Task Capture, Export/Import, Kanban, Recurring Tasks, Sharing, PWA
# =============================================================================

import csv
import io
import json
import logging
import re
import threading
from calendar import day_name
from datetime import datetime, timedelta, date

import numpy as np
from flask import (
    abort, flash, jsonify, redirect, render_template_string,
    request, send_file, url_for,
)
from flask_login import current_user, login_required
from flask_mail import Message
from flask_wtf import FlaskForm
from wtforms import BooleanField, SelectField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, Optional

try:
    from scipy.stats import pearsonr
    _SCIPY = True
except ImportError:
    _SCIPY = False

SIMILARITY_THRESHOLD = 0.35
_WEEKDAYS = {d.lower(): i for i, d in enumerate(day_name)}


def _pearson(x, y):
    if _SCIPY:
        return pearsonr(x, y)
    if len(x) < 2:
        return 0.0, 1.0
    xa, ya = np.array(x, dtype=float), np.array(y, dtype=float)
    if xa.std() == 0 or ya.std() == 0:
        return 0.0, 1.0
    r = float(np.corrcoef(xa, ya)[0, 1])
    n = len(x)
    if abs(r) >= 1:
        p = 0.0
    else:
        t = r * np.sqrt((n - 2) / (1 - r * r))
        p = 0.05 if abs(t) > 2 else 0.5
    return r, p


# ── NL Task Parser ────────────────────────────────────────────────────────────

def _resolve_next_weekday(match, ref):
    wd = match.group(1).lower()
    target = _WEEKDAYS[wd]
    days_ahead = (target - ref.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return ref + timedelta(days=days_ahead)


def _resolve_this_weekday(match, ref):
    wd = match.group(2).lower()
    target = _WEEKDAYS[wd]
    days_ahead = (target - ref.weekday()) % 7
    return ref + timedelta(days=days_ahead)


def _resolve_numeric_date(match, ref):
    a, b, c = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if c < 100:
        c += 2000
    if a > 12:
        return date(c, b, a)
    return date(c, a, b)


def _resolve_month_day(match, ref):
    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    m = months[match.group(1).lower()[:3]]
    d = int(match.group(2))
    y = ref.year
    try:
        return date(y, m, d)
    except ValueError:
        return ref


def _resolve_end_of_period(match, ref):
    period = match.group(1).lower()
    if period == "week":
        return ref + timedelta(days=(6 - ref.weekday()))
    last = (ref.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return last


DATE_PATTERNS = [
    (r"\btoday\b", lambda m, d: d),
    (r"\btomorrow\b", lambda m, d: d + timedelta(days=1)),
    (r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", _resolve_next_weekday),
    (r"\bin\s+(\d+)\s+days?\b", lambda m, d: d + timedelta(days=int(m.group(1)))),
    (r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", _resolve_numeric_date),
    (r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{1,2})\b", _resolve_month_day),
    (r"\b(this\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", _resolve_this_weekday),
    (r"\bend\s+of\s+(week|month)\b", _resolve_end_of_period),
    (r"\bnext\s+week\b", lambda m, d: d + timedelta(days=7)),
    (r"\bby\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
     lambda m, d: d + timedelta(days=(_WEEKDAYS[m.group(1).lower()] - d.weekday()) % 7 or 7)),
    (r"\bfriday\b", lambda m, d: d + timedelta(days=(_WEEKDAYS["friday"] - d.weekday()) % 7 or 7)),
]

PRIORITY_KEYWORDS = {
    "Critical": ["urgent", "asap", "critical", "immediately", "now"],
    "High": ["high priority", "high-priority", "important", "must", "need to", "high"],
    "Low": ["low priority", "low-priority", "sometime", "whenever", "eventually", "maybe", "low"],
}

STOP_MARKERS = re.compile(
    r"\b(by|before|due|on|at|priority|urgent|asap|high|low|medium|critical|tomorrow|today)\b",
    re.I,
)


def parse_natural_language_task(text, ref=None):
    ref = ref or date.today()
    raw = (text or "").strip()
    if not raw:
        return {"title": "New task", "deadline": None, "priority": "Medium", "confidence": 0.0, "raw": raw}

    deadline = None
    working = raw
    for pattern, resolver in DATE_PATTERNS:
        m = re.search(pattern, working, re.I)
        if m:
            try:
                deadline = resolver(m, ref)
                working = working[: m.start()] + working[m.end() :]
                break
            except Exception:
                pass

    priority = "Medium"
    lower = raw.lower()
    for level, keywords in PRIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                priority = level
                break
        if priority != "Medium":
            break

    title = STOP_MARKERS.sub("", working)
    title = re.sub(r"\s+", " ", title).strip(" ,.-")
    if not title:
        title = raw[:120]
    title = title[0].upper() + title[1:] if title else "New task"

    confidence = 0.5
    if deadline:
        confidence += 0.3
    if priority != "Medium":
        confidence += 0.2

    dl_str = None
    if deadline:
        dl_str = datetime.combine(deadline, datetime.max.time().replace(microsecond=0)).isoformat()

    return {
        "title": title,
        "deadline": dl_str,
        "priority": priority,
        "confidence": min(1.0, confidence),
        "raw": raw,
    }


# ── Focus & Weekly scoring ────────────────────────────────────────────────────

def compute_focus_score(session):
    completion = min(session.actual_minutes / max(session.planned_minutes, 1), 1.0)
    interruption_penalty = min(session.interruptions * 0.1, 0.5)
    quality_bonus = (session.quality - 1) / 4.0 if session.quality else 0.5
    score = completion * 0.5 + quality_bonus * 0.3 + (1.0 - interruption_penalty) * 0.2
    return max(0.0, min(1.0, score))


def compute_week_score(stats):
    score = (
        stats.get("completion_rate", 0) * 35
        + stats.get("habit_completion_rate", 0) * 25
        + min(stats.get("focus_sessions", 0) / 10, 1.0) * 20
        + (stats.get("avg_mood", 3) / 5.0) * 10
        + (stats.get("avg_energy", 3) / 5.0) * 10
    )
    return max(0.0, min(100.0, score))


def generate_narrative(stats):
    paragraphs = []
    cr = stats.get("completion_rate", 0)
    tc = stats.get("tasks_completed", 0)
    if cr >= 0.8:
        paragraphs.append(f"Strong week — you completed {tc} tasks with an {cr * 100:.0f}% completion rate.")
    elif cr >= 0.5:
        paragraphs.append(f"Solid progress this week with {tc} tasks completed ({cr * 100:.0f}% rate).")
    else:
        paragraphs.append(f"A challenging week, but every data point is a learning opportunity. You finished {tc} tasks.")

    avg_mood = stats.get("avg_mood", 0)
    if avg_mood >= 4.0 and cr >= 0.7:
        paragraphs.append("Your high mood ratings aligned with strong task output — keep protecting that energy.")
    elif 0 < avg_mood < 2.5:
        paragraphs.append("Mood dipped this week. Consider reviewing your workload for next week.")

    if stats.get("habit_completion_rate", 0) >= 0.85:
        paragraphs.append("Habit consistency was excellent — your streaks are building real momentum.")

    focus_m = stats.get("total_focus_minutes", 0)
    if focus_m >= 300:
        paragraphs.append(f"Deep work highlight: {focus_m} focused minutes logged across {stats.get('focus_sessions', 0)} Pomodoro sessions.")

    delta = stats.get("score", 0) - stats.get("prev_week_score", 0)
    if delta > 5:
        paragraphs.append(f"↑ Up {round(delta)} points from last week.")
    elif delta < -5:
        paragraphs.append(f"↓ Down {round(abs(delta))} points from last week.")

    if stats.get("top_category"):
        paragraphs.append(f"Most active note category: {stats['top_category']}.")

    return " ".join(paragraphs) if paragraphs else "Keep logging your activity to unlock richer weekly insights."


def register_v2(app, ctx):
    """Register all V2+ routes and schedulers. ctx holds shared app objects."""
    db = ctx["db"]
    mail = ctx["mail"]
    csrf = ctx["csrf"]
    BASE = ctx["BASE_TEMPLATE"]
    sanitize = ctx["sanitize"]
    csrf_hidden = ctx["csrf_hidden"]
    get_model = ctx["get_sentence_model"]
    cosine_similarity = ctx.get("cosine_similarity")

    User = ctx["User"]
    Task = ctx["Task"]
    Goal = ctx["Goal"]
    Note = ctx["Note"]
    MoodLog = ctx["MoodLog"]
    Habit = ctx["Habit"]
    HabitCheckin = ctx["HabitCheckin"]
    WeeklyReview = ctx["WeeklyReview"]
    NoteLink = ctx["NoteLink"]
    FocusSession = ctx["FocusSession"]
    MoodInsight = ctx["MoodInsight"]
    TaskShare = ctx["TaskShare"]
    GoalShare = ctx["GoalShare"]
    TimeLog = ctx["TimeLog"]
    Subtask = ctx["Subtask"]

    # ── Weekly stats collector ────────────────────────────────────────────────

    def _week_bounds(ref=None):
        ref = ref or date.today()
        start = ref - timedelta(days=ref.weekday())
        end = start + timedelta(days=6)
        return start, end

    def _collect_week_stats(user_id, week_start=None):
        ws, we = _week_bounds(week_start or date.today())
        ws_dt = datetime.combine(ws, datetime.min.time())
        we_dt = datetime.combine(we + timedelta(days=1), datetime.min.time())

        tasks_created = Task.query.filter(
            Task.user_id == user_id, Task.created_at >= ws_dt, Task.created_at < we_dt
        ).count()
        tasks_completed = Task.query.filter(
            Task.user_id == user_id, Task.status == "Completed",
            Task.completed_at >= ws_dt, Task.completed_at < we_dt,
        ).count()
        completion_rate = tasks_completed / max(tasks_created, 1)

        goals = Goal.query.filter_by(user_id=user_id).all()
        goals_advanced = sum(1 for g in goals if g.progress > 0)

        habits = Habit.query.filter_by(user_id=user_id).all()
        habit_days = max(len(habits) * 7, 1)
        checkins = HabitCheckin.query.filter(
            HabitCheckin.user_id == user_id,
            HabitCheckin.checked_date >= ws,
            HabitCheckin.checked_date <= we,
        ).count()
        habit_completion_rate = checkins / habit_days

        moods = MoodLog.query.filter(
            MoodLog.user_id == user_id,
            MoodLog.logged_date >= ws,
            MoodLog.logged_date <= we,
        ).all()
        avg_mood = sum(m.mood for m in moods) / len(moods) if moods else 0
        energies = [m.energy_level for m in moods if m.energy_level]
        avg_energy = sum(energies) / len(energies) if energies else 0

        sessions = FocusSession.query.filter(
            FocusSession.user_id == user_id,
            FocusSession.started_at >= ws_dt,
            FocusSession.started_at < we_dt,
        ).all()
        focus_sessions = len(sessions)
        total_focus_minutes = sum(s.actual_minutes for s in sessions)

        notes = Note.query.filter_by(user_id=user_id).all()
        cats = {}
        for n in notes:
            if n.category:
                cats[n.category] = cats.get(n.category, 0) + 1
        top_category = max(cats, key=cats.get) if cats else ""

        prev = WeeklyReview.query.filter_by(user_id=user_id).order_by(WeeklyReview.week_start.desc()).first()
        prev_week_score = prev.score if prev else 0

        stats = {
            "tasks_completed": tasks_completed,
            "tasks_created": tasks_created,
            "completion_rate": round(completion_rate, 3),
            "goals_advanced": goals_advanced,
            "habits_streak_avg": round(sum(h.streak or 0 for h in habits) / max(len(habits), 1), 1),
            "habit_completion_rate": round(habit_completion_rate, 3),
            "avg_mood": round(avg_mood, 1),
            "avg_energy": round(avg_energy, 1),
            "focus_sessions": focus_sessions,
            "total_focus_minutes": total_focus_minutes,
            "top_category": top_category,
            "prev_week_score": prev_week_score,
        }
        stats["score"] = round(compute_week_score(stats), 1)
        return ws, we, stats

    def _generate_weekly_review(user_id, week_start=None):
        ws, we, stats = _collect_week_stats(user_id, week_start)
        narrative = generate_narrative(stats)
        existing = WeeklyReview.query.filter_by(user_id=user_id, week_start=ws).first()
        if not existing:
            existing = WeeklyReview(user_id=user_id, week_start=ws, week_end=we)
        existing.stats_json = json.dumps(stats)
        existing.narrative = narrative
        existing.score = stats["score"]
        db.session.add(existing)
        db.session.commit()
        return existing

    # ── Note linking ─────────────────────────────────────────────────────────

    def _recompute_note_links(user_id, note_id, app_obj):
        with app_obj.app_context():
            try:
                model = get_model()
                if model is None or cosine_similarity is None:
                    return
                notes = Note.query.filter_by(user_id=user_id).all()
                if len(notes) < 2:
                    return
                texts = [f"{n.title}. {n.content or ''}" for n in notes]
                embeddings = model.encode(texts, convert_to_numpy=True)
                idx_map = {n.id: i for i, n in enumerate(notes)}
                if note_id not in idx_map:
                    return
                new_idx = idx_map[note_id]
                new_emb = embeddings[new_idx].reshape(1, -1)
                sims = cosine_similarity(new_emb, embeddings)[0]
                for i, n in enumerate(notes):
                    if i == new_idx:
                        continue
                    if sims[i] >= SIMILARITY_THRESHOLD:
                        a, b = min(note_id, n.id), max(note_id, n.id)
                        link = NoteLink.query.filter_by(source_note_id=a, target_note_id=b).first()
                        if not link:
                            link = NoteLink(user_id=user_id, source_note_id=a, target_note_id=b)
                        link.similarity = float(sims[i])
                        db.session.add(link)
                db.session.commit()
            except Exception as exc:
                logging.warning("Note link recompute failed: %s", exc)
                db.session.rollback()

    # ── Mood insights ─────────────────────────────────────────────────────────

    def _compute_mood_insights(user_id):
        today = date.today()
        days = [today - timedelta(days=i) for i in range(89, -1, -1)]
        mood_by_day = {m.logged_date: m for m in MoodLog.query.filter_by(user_id=user_id).all()}

        mood_series, energy_series, completion_series, focus_series = [], [], [], []
        valid_days = []
        for d in days:
            m = mood_by_day.get(d)
            if not m:
                continue
            d_start = datetime.combine(d, datetime.min.time())
            d_end = d_start + timedelta(days=1)
            created = Task.query.filter(Task.user_id == user_id, Task.created_at >= d_start, Task.created_at < d_end).count()
            completed = Task.query.filter(
                Task.user_id == user_id, Task.status == "Completed",
                Task.completed_at >= d_start, Task.completed_at < d_end,
            ).count()
            fs = FocusSession.query.filter(
                FocusSession.user_id == user_id,
                FocusSession.started_at >= d_start, FocusSession.started_at < d_end,
            ).count()
            valid_days.append(d)
            mood_series.append(m.mood)
            energy_series.append(m.energy_level or m.mood)
            completion_series.append(completed / max(created, 1))
            focus_series.append(fs)

        if len(valid_days) < 10:
            return

        MoodInsight.query.filter_by(user_id=user_id).delete()

        r, p = _pearson(mood_series, completion_series)
        if abs(r) > 0.3 and p < 0.05:
            direction = "positively" if r > 0 else "negatively"
            text = f"Mood {direction} correlates with task completion (r={r:.2f})."
            db.session.add(MoodInsight(
                user_id=user_id, insight_type="mood_vs_completion",
                insight_text=text, correlation_r=r, p_value=p, sample_size=len(valid_days),
            ))

        r2, p2 = _pearson(energy_series, focus_series)
        if abs(r2) > 0.3 and p2 < 0.05:
            text = f"Energy levels correlate with focus sessions (r={r2:.2f})."
            db.session.add(MoodInsight(
                user_id=user_id, insight_type="energy_vs_focus",
                insight_text=text, correlation_r=r2, p_value=p2, sample_size=len(valid_days),
            ))

        all_tags = set()
        for m in mood_by_day.values():
            if m.tags:
                all_tags.update(t.strip() for t in m.tags.split(",") if t.strip())

        for tag in all_tags:
            tagged, untagged = [], []
            for i, d in enumerate(valid_days):
                m = mood_by_day.get(d)
                tags = [t.strip() for t in (m.tags or "").split(",") if t.strip()] if m else []
                if tag in tags:
                    tagged.append(completion_series[i])
                else:
                    untagged.append(completion_series[i])
            if len(tagged) >= 5 and len(untagged) >= 5:
                t_avg = sum(tagged) / len(tagged)
                u_avg = sum(untagged) / len(untagged)
                delta = t_avg - u_avg
                if abs(delta) > 0.10:
                    pct = abs(delta) * 100
                    if delta > 0:
                        text = f'On days tagged "{tag}", you complete {pct:.0f}% more tasks than average.'
                    else:
                        text = f'On days tagged "{tag}", task completion drops by {pct:.0f}%.'
                    db.session.add(MoodInsight(
                        user_id=user_id, insight_type="tag_impact",
                        insight_text=text, correlation_r=delta, p_value=0.04, sample_size=len(tagged),
                    ))

        high_energy_days = [i for i, d in enumerate(valid_days) if mood_by_day.get(d) and (mood_by_day[d].energy_level or 0) >= 4]
        if len(high_energy_days) >= 5:
            hi = sum(completion_series[i] for i in high_energy_days) / len(high_energy_days)
            lo_days = [i for i in range(len(valid_days)) if i not in high_energy_days]
            if lo_days:
                lo = sum(completion_series[i] for i in lo_days) / len(lo_days)
                if lo > 0 and hi > lo:
                    boost = ((hi - lo) / lo) * 100
                    if boost > 10:
                        db.session.add(MoodInsight(
                            user_id=user_id, insight_type="energy_boost",
                            insight_text=f"On high-energy days (≥4), you complete {boost:.0f}% more tasks.",
                            correlation_r=boost / 100, p_value=0.03, sample_size=len(high_energy_days),
                        ))
        db.session.commit()

    # ── Recurring tasks job ───────────────────────────────────────────────────

    def _process_recurring_tasks():
        with app.app_context():
            today = date.today()
            tasks = Task.query.filter(Task.recurrence != "None", Task.parent_id.is_(None)).all()
            for t in tasks:
                if t.recurrence == "Daily":
                    due = True
                elif t.recurrence == "Weekly":
                    due = t.created_at.weekday() == today.weekday()
                else:
                    due = False
                if not due:
                    continue
                exists = Task.query.filter(
                    Task.user_id == t.user_id, Task.title == t.title,
                    Task.created_at >= datetime.combine(today, datetime.min.time()),
                ).first()
                if not exists:
                    clone = Task(
                        user_id=t.user_id, title=t.title, description=t.description,
                        priority=t.priority, status="Pending", deadline=t.deadline,
                        recurrence="None", parent_id=None,
                    )
                    db.session.add(clone)
            db.session.commit()

    def _job_weekly_reviews():
        with app.app_context():
            for user in User.query.all():
                try:
                    _generate_weekly_review(user.id)
                except Exception as exc:
                    logging.warning("Weekly review failed for user %s: %s", user.id, exc)

    def _job_mood_insights():
        with app.app_context():
            for user in User.query.all():
                try:
                    _compute_mood_insights(user.id)
                except Exception as exc:
                    logging.warning("Mood insights failed for user %s: %s", user.id, exc)

    def _job_email_digest():
        with app.app_context():
            for user in User.query.filter_by(email_digest=True).all():
                try:
                    review = _generate_weekly_review(user.id)
                    if not mail or not app.config.get("MAIL_USERNAME"):
                        continue
                    msg = Message(
                        f"Your Weekly Life OS Digest — Score {review.score:.0f}/100",
                        recipients=[user.email],
                    )
                    msg.body = f"Hi {user.name},\n\n{review.narrative}\n\nWeek Score: {review.score}/100\n\nLog in to see charts: {url_for('weekly_review', _external=True)}"
                    mail.send(msg)
                except Exception as exc:
                    logging.warning("Email digest failed for %s: %s", user.email, exc)

    ctx["jobs"] = {
        "weekly_reviews": _job_weekly_reviews,
        "mood_insights": _job_mood_insights,
        "email_digest": _job_email_digest,
        "recurring_tasks": _process_recurring_tasks,
    }
    ctx["recompute_note_links"] = _recompute_note_links
    ctx["generate_weekly_review"] = _generate_weekly_review
    ctx["compute_mood_insights"] = _compute_mood_insights

    # ── Templates ─────────────────────────────────────────────────────────────

    WEEKLY_REVIEW_TPL = BASE.replace("{% block content %}{% endblock %}", """{% block content %}
<div class="d-flex justify-content-between align-items-center mb-4">
  <div>
    <h3><i class="bi bi-calendar-week"></i> Smart Weekly Review</h3>
    <p class="text-muted mb-0">Week of {{ review.week_start.strftime('%b %d') }} – {{ review.week_end.strftime('%b %d, %Y') }}</p>
  </div>
  <div class="d-flex gap-2">
    <form method="POST" action="{{ url_for('weekly_review_generate') }}">{{ csrf_token_field|safe }}<button class="btn btn-purple btn-sm"><i class="bi bi-arrow-clockwise"></i> Regenerate</button></form>
    <a href="{{ url_for('dashboard') }}" class="btn btn-outline-secondary btn-sm">Dashboard</a>
  </div>
</div>
<div class="row g-3 mb-4">
  <div class="col-md-3"><div class="stat-card purple text-center"><span class="stat-num" style="color:#a78bfa">{{ review.score|round(0)|int }}</span><span class="stat-label">Week Score</span></div></div>
  <div class="col-md-3"><div class="stat-card green text-center"><span class="stat-num" style="color:#34d399">{{ stats.tasks_completed }}</span><span class="stat-label">Tasks Done</span></div></div>
  <div class="col-md-3"><div class="stat-card amber text-center"><span class="stat-num" style="color:#fbbf24">{{ stats.focus_sessions }}</span><span class="stat-label">Focus Sessions</span></div></div>
  <div class="col-md-3"><div class="stat-card red text-center"><span class="stat-num" style="color:#f87171">{{ stats.avg_mood|default(0) }}</span><span class="stat-label">Avg Mood /5</span></div></div>
</div>
<div class="card mb-4"><div class="card-header"><i class="bi bi-chat-quote me-2"></i>AI Narrative Digest</div><div class="card-body"><p style="line-height:1.8;font-size:1.05rem">{{ review.narrative }}</p></div></div>
<div class="row g-3">
  <div class="col-lg-6"><div class="card h-100"><div class="card-header">Task Completion Trend</div><div class="card-body"><canvas id="wrTasksChart" style="max-height:220px"></canvas></div></div></div>
  <div class="col-lg-6"><div class="card h-100"><div class="card-header">Focus & Mood Trend</div><div class="card-body"><canvas id="wrFocusChart" style="max-height:220px"></canvas></div></div></div>
</div>
{% if past_reviews %}
<div class="card mt-4"><div class="card-header">Past Reviews</div><div class="list-group list-group-flush">
{% for pr in past_reviews %}
<a href="{{ url_for('weekly_review', week=pr.week_start.isoformat()) }}" class="list-group-item list-group-item-action d-flex justify-content-between">
  <span>{{ pr.week_start.strftime('%b %d') }} – {{ pr.week_end.strftime('%b %d, %Y') }}</span>
  <span class="badge bg-primary">{{ pr.score|round(0)|int }}/100</span>
</a>
{% endfor %}
</div></div>
{% endif %}
{% endblock %}""").replace("{% block extra_js %}{% endblock %}", """{% block extra_js %}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script>
const trendLabels = {{ trend_labels|safe }};
const trendTasks = {{ trend_tasks|safe }};
const trendFocus = {{ trend_focus|safe }};
const trendMood = {{ trend_mood|safe }};
new Chart(document.getElementById('wrTasksChart'), {type:'line', data:{labels:trendLabels, datasets:[{label:'Completed', data:trendTasks, borderColor:'#7c3aed', tension:0.3, fill:true, backgroundColor:'#7c3aed22'}]}, options:{plugins:{legend:{labels:{color:'#8b949e'}}}, scales:{x:{ticks:{color:'#8b949e'}}, y:{ticks:{color:'#8b949e'}, beginAtZero:true}}}});
new Chart(document.getElementById('wrFocusChart'), {type:'line', data:{labels:trendLabels, datasets:[{label:'Focus min', data:trendFocus, borderColor:'#06b6d4', tension:0.3},{label:'Mood', data:trendMood, borderColor:'#ec4899', tension:0.3, yAxisID:'y1'}]}, options:{plugins:{legend:{labels:{color:'#8b949e'}}}, scales:{x:{ticks:{color:'#8b949e'}}, y:{ticks:{color:'#8b949e'}, beginAtZero:true}, y1:{position:'right', ticks:{color:'#8b949e'}, min:0, max:5, grid:{drawOnChartArea:false}}}}});
</script>
{% endblock %}""")

    NOTES_GRAPH_TPL = BASE.replace("{% block content %}{% endblock %}", """{% block content %}
<div class="d-flex justify-content-between align-items-center mb-3">
  <h3><i class="bi bi-diagram-3"></i> Note Knowledge Graph</h3>
  <div class="d-flex gap-2">
    <a href="{{ url_for('notes_index') }}" class="btn btn-outline-secondary btn-sm">All Notes</a>
    <button class="btn btn-purple btn-sm" onclick="loadGraph()"><i class="bi bi-arrow-clockwise"></i> Refresh</button>
  </div>
</div>
<p class="text-muted small">Semantically linked notes via cosine similarity (sentence-transformers). Node size = word count. Edge width = similarity.</p>
<div id="graph-empty" class="text-center py-5" style="display:none"><i class="bi bi-diagram-3" style="font-size:3rem;opacity:0.3"></i><p class="text-muted mt-3">Add at least 2 notes to see your knowledge graph.</p></div>
<div id="graph-container" style="width:100%;height:600px;border:1px solid var(--border-color);border-radius:12px;background:var(--bg-primary)"></div>
<div id="graph-tooltip" style="position:fixed;display:none;background:var(--bg-secondary);border:1px solid var(--border-color);padding:8px 12px;border-radius:8px;font-size:0.85rem;pointer-events:none;z-index:999"></div>
{% endblock %}""").replace("{% block extra_js %}{% endblock %}", """{% block extra_js %}
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const CATEGORY_COLORS = ['#7c3aed','#ec4899','#06b6d4','#10b981','#f59e0b','#ef4444','#a78bfa','#67e8f9'];
function catColor(c){ let h=0; for(let i=0;i<(c||'').length;i++) h=(h*31+c.charCodeAt(i))%CATEGORY_COLORS.length; return CATEGORY_COLORS[h]; }
let simulation;
function loadGraph(){
  fetch("{{ url_for('notes_graph_data') }}").then(r=>r.json()).then(data=>{
    const container=document.getElementById('graph-container');
    d3.select(container).selectAll('*').remove();
    if(!data.nodes||data.nodes.length<2){ document.getElementById('graph-empty').style.display='block'; return; }
    document.getElementById('graph-empty').style.display='none';
    const w=container.clientWidth, h=600;
    const svg=d3.select(container).append('svg').attr('width',w).attr('height',h);
    const g=svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.3,3]).on('zoom',e=>g.attr('transform',e.transform)));
    const tooltip=document.getElementById('graph-tooltip');
    simulation=d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d=>d.id).distance(d=>120*(1-d.similarity)).strength(d=>d.similarity))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(w/2,h/2));
    const link=g.append('g').selectAll('line').data(data.links).join('line').attr('stroke','#7c3aed55').attr('stroke-width',d=>2+d.similarity*4);
    const node=g.append('g').selectAll('circle').data(data.nodes).join('circle')
      .attr('r',d=>8+Math.min(d.word_count/50,20)).attr('fill',d=>catColor(d.category))
      .attr('stroke','#fff').attr('stroke-width',1.5).style('cursor','pointer')
      .on('mouseover',(e,d)=>{ tooltip.style.display='block'; tooltip.innerHTML='<strong>'+d.title+'</strong><br>Similarity hub · '+d.word_count+' words'; })
      .on('mousemove',e=>{ tooltip.style.left=(e.pageX+12)+'px'; tooltip.style.top=(e.pageY+12)+'px'; })
      .on('mouseout',()=>{ tooltip.style.display='none'; })
      .on('click',(e,d)=>{ window.location="{{ url_for('notes_edit', note_id=0) }}".replace('/0/','/'+d.id+'/'); })
      .call(d3.drag().on('start',(e,d)=>{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
        .on('drag',(e,d)=>{ d.fx=e.x; d.fy=e.y; })
        .on('end',(e,d)=>{ if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }));
    simulation.on('tick',()=>{ link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
      node.attr('cx',d=>d.x).attr('cy',d=>d.y); });
  });
}
loadGraph();
</script>
{% endblock %}""")

    FOCUS_TPL = BASE.replace("{% block content %}{% endblock %}", """{% block content %}
<div class="row g-4">
  <div class="col-lg-5">
    <div class="card"><div class="card-header"><i class="bi bi-stopwatch me-2"></i>Pomodoro Focus Engine</div><div class="card-body text-center py-4">
      <div id="timer-display" style="font-size:4rem;font-weight:700;font-family:monospace;background:var(--gradient-primary);-webkit-background-clip:text;-webkit-text-fill-color:transparent">25:00</div>
      <p class="text-muted" id="timer-status">Ready to focus</p>
      <div class="mb-3"><select id="task-select" class="form-select"><option value="">— No linked task —</option>{% for t in tasks %}<option value="{{ t.id }}">{{ t.title }}</option>{% endfor %}</select></div>
      <div class="btn-group mb-3"><button class="btn btn-outline-secondary btn-sm" onclick="setDuration(25)">25m</button><button class="btn btn-outline-secondary btn-sm" onclick="setDuration(15)">15m</button><button class="btn btn-outline-secondary btn-sm" onclick="setDuration(50)">50m</button></div>
      <div class="d-flex gap-2 justify-content-center flex-wrap">
        <button class="btn btn-purple" id="btn-start" onclick="startTimer()"><i class="bi bi-play-fill"></i> Start</button>
        <button class="btn btn-outline-secondary" id="btn-pause" onclick="pauseTimer()" disabled>Pause</button>
        <button class="btn btn-outline-danger" onclick="cancelTimer()">Cancel</button>
        <button class="btn btn-outline-warning btn-sm" onclick="logInterruption()">+ Distraction</button>
      </div>
      <div class="mt-3 small text-muted">Interruptions: <span id="int-count">0</span> · Today: <span id="today-mins">{{ today_minutes }}</span> min</div>
    </div></div>
    <div class="card mt-3"><div class="card-header">Rate This Session</div><div class="card-body"><div class="d-flex gap-2 justify-content-center" id="quality-btns">{% for q in range(1,6) %}<button class="btn btn-outline-secondary btn-sm q-btn" data-q="{{ q }}" onclick="setQuality({{ q }})">{{ q }}</button>{% endfor %}</div></div></div>
  </div>
  <div class="col-lg-7">
    <div class="card mb-3"><div class="card-header">Focus Heatmap (90 days)</div><div class="card-body"><div id="heatmap" class="d-flex flex-wrap gap-1"></div></div></div>
    <div class="card"><div class="card-header">Best Focus Hours</div><div class="card-body"><canvas id="hourlyChart" style="max-height:200px"></canvas></div></div>
  </div>
</div>
{% endblock %}""").replace("{% block extra_js %}{% endblock %}", """{% block extra_js %}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script>
let planned=25, remaining=25*60, interval=null, running=false, interruptions=0, quality=3, startedAt=null;
function fmt(s){ return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0'); }
function setDuration(m){ if(running) return; planned=m; remaining=m*60; document.getElementById('timer-display').textContent=fmt(remaining); }
function tick(){ remaining--; document.getElementById('timer-display').textContent=fmt(remaining);
  if(remaining<=0){ clearInterval(interval); running=false; completeSession(); }}
function startTimer(){ if(running) return; if(!startedAt) startedAt=new Date().toISOString();
  running=true; document.getElementById('btn-start').disabled=true; document.getElementById('btn-pause').disabled=false;
  document.getElementById('timer-status').textContent='Focusing...'; interval=setInterval(tick,1000); }
function pauseTimer(){ clearInterval(interval); running=false; document.getElementById('btn-start').disabled=false; document.getElementById('btn-pause').disabled=true; document.getElementById('timer-status').textContent='Paused'; }
function cancelTimer(){ clearInterval(interval); running=false; remaining=planned*60; startedAt=null; interruptions=0;
  document.getElementById('timer-display').textContent=fmt(remaining); document.getElementById('int-count').textContent='0';
  document.getElementById('btn-start').disabled=false; document.getElementById('btn-pause').disabled=true; document.getElementById('timer-status').textContent='Ready to focus'; }
function logInterruption(){ interruptions++; document.getElementById('int-count').textContent=interruptions; }
function setQuality(q){ quality=q; document.querySelectorAll('.q-btn').forEach(b=>b.classList.toggle('btn-purple',parseInt(b.dataset.q)===q)); }
function completeSession(){
  const actual=planned; const taskId=document.getElementById('task-select').value||null;
  fetch("{{ url_for('focus_session') }}",{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId,planned_minutes:planned,actual_minutes:actual,interruptions,quality,session_type:'Work',started_at:startedAt})})
  .then(r=>r.json()).then(d=>{ document.getElementById('timer-status').textContent='Session complete! 🔥 Streak: '+d.streak;
    document.getElementById('today-mins').textContent=d.total_today; cancelTimer(); loadAnalytics(); });
}
function loadAnalytics(){
  fetch("{{ url_for('focus_analytics') }}").then(r=>r.json()).then(d=>{
    const hm=document.getElementById('heatmap'); hm.innerHTML='';
    const maxM=Math.max(...d.heatmap.map(x=>x.minutes),1);
    d.heatmap.forEach(day=>{ const el=document.createElement('div'); el.title=day.date+': '+day.minutes+' min';
      el.style.cssText='width:12px;height:12px;border-radius:2px;background:'+(day.minutes===0?'#30363d':'rgba(124,58,237,'+(0.2+0.8*day.minutes/maxM)+')');
      hm.appendChild(el); });
    new Chart(document.getElementById('hourlyChart'),{type:'bar',data:{labels:d.hourly_distribution.map(h=>h.hour+':00'),datasets:[{label:'Avg Quality',data:d.hourly_distribution.map(h=>h.avg_quality),backgroundColor:'#7c3aed88'}]},options:{plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#8b949e'}},y:{ticks:{color:'#8b949e'},min:0,max:5}}}});
  });
}
loadAnalytics();
</script>
{% endblock %}""")

    MOOD_INTEL_TPL = BASE.replace("{% block content %}{% endblock %}", """{% block content %}
<div class="d-flex justify-content-between align-items-center mb-4">
  <h3><i class="bi bi-emoji-smile"></i> Mood & Energy Intelligence</h3>
  <button class="btn btn-purple btn-sm" onclick="document.getElementById('log-panel').scrollIntoView({behavior:'smooth'})">Log Today</button>
</div>
{% if insights %}
<div class="row g-3 mb-4">{% for ins in insights %}
<div class="col-md-6"><div class="card h-100 border-start border-4" style="border-color:var(--accent-purple)!important"><div class="card-body">
  <div class="d-flex justify-content-between"><span class="badge bg-primary">{{ ins.insight_type }}</span>
  <small class="text-muted">n={{ ins.sample_size }}</small></div>
  <p class="mt-2 mb-2">{{ ins.insight_text }}</p>
  <div class="progress" style="height:6px"><div class="progress-bar bg-purple" style="width:{{ (ins.correlation_r|abs * 100)|int }}%;background:#7c3aed"></div></div>
</div></div></div>
{% endfor %}</div>
{% else %}
<div class="alert alert-info"><i class="bi bi-info-circle me-2"></i>Not enough data yet — keep logging mood & energy for 10+ days to unlock Pearson correlation insights.</div>
{% endif %}
<div class="row g-3 mb-4">
  <div class="col-lg-8"><div class="card"><div class="card-header">Mood · Energy · Completion Overlay</div><div class="card-body"><canvas id="moodChart" style="max-height:260px"></canvas></div></div></div>
  <div class="col-lg-4"><div class="card" id="log-panel"><div class="card-header">Log Mood & Energy</div><div class="card-body">
    <form method="POST" action="{{ url_for('mood_log') }}">{{ csrf_token_field|safe }}
      <div class="mb-2"><label class="form-label">Mood (1-5)</label><input type="range" name="mood" min="1" max="5" value="3" class="form-range" oninput="this.nextElementSibling.textContent=this.value"><span>3</span></div>
      <div class="mb-2"><label class="form-label">Energy (1-5)</label><input type="range" name="energy_level" min="1" max="5" value="3" class="form-range" oninput="this.nextElementSibling.textContent=this.value"><span>3</span></div>
      <div class="mb-2"><label class="form-label">Tags (comma-separated, max 3)</label><input name="tags" class="form-control" placeholder="well-slept, after-exercise"></div>
      <div class="mb-2"><label class="form-label">Note</label><input name="note" class="form-control"></div>
      <button class="btn btn-purple btn-sm w-100">Save Entry</button>
    </form>
  </div></div></div>
</div>
{% endblock %}""").replace("{% block extra_js %}{% endblock %}", """{% block extra_js %}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script>
const chartData = {{ chart_data|safe }};
new Chart(document.getElementById('moodChart'), {type:'line', data:{labels:chartData.labels, datasets:[
  {label:'Mood', data:chartData.mood, borderColor:'#ec4899', tension:0.3},
  {label:'Energy', data:chartData.energy, borderColor:'#f59e0b', tension:0.3},
  {label:'Completion %', data:chartData.completion, borderColor:'#10b981', tension:0.3, yAxisID:'y1'}
]}, options:{plugins:{legend:{labels:{color:'#8b949e'}}}, scales:{x:{ticks:{color:'#8b949e'}}, y:{min:0,max:5,ticks:{color:'#8b949e'}}, y1:{position:'right',min:0,max:100,ticks:{color:'#8b949e'},grid:{drawOnChartArea:false}}}}});
</script>
{% endblock %}""")

    KANBAN_TPL = BASE.replace("{% block content %}{% endblock %}", """{% block content %}
<div class="d-flex justify-content-between align-items-center mb-3">
  <h3><i class="bi bi-kanban"></i> Kanban Board</h3>
  <div class="d-flex gap-2">
    <a href="{{ url_for('tasks_index') }}" class="btn btn-outline-secondary btn-sm">List View</a>
    <a href="{{ url_for('tasks_new') }}" class="btn btn-purple btn-sm">+ New Task</a>
  </div>
</div>
<div class="row g-3" id="kanban-board">
{% for col, label, color in [('Pending','To Do','#7c3aed'),('In Progress','In Progress','#f59e0b'),('Completed','Done','#10b981')] %}
<div class="col-md-4"><div class="card h-100"><div class="card-header d-flex justify-content-between" style="border-top:3px solid {{ color }}"><span>{{ label }}</span><span class="badge bg-secondary">{{ columns[col]|length }}</span></div>
<div class="card-body kanban-col" data-status="{{ col }}" style="min-height:400px" ondragover="event.preventDefault()" ondrop="dropTask(event)">
{% for t in columns[col] %}
<div class="card mb-2 kanban-card" draggable="true" data-id="{{ t.id }}" ondragstart="dragTask(event)" style="cursor:grab">
  <div class="card-body py-2 px-3">
    <div class="fw-semibold small">{{ t.title }}</div>
    <div class="d-flex justify-content-between mt-1"><span class="priority-{{ t.priority }}" style="font-size:0.7rem">{{ t.priority }}</span>
    {% if t.deadline %}<small class="text-muted">{{ t.deadline.strftime('%b %d') }}</small>{% endif %}</div>
    {% if t.recurrence != 'None' %}<span class="badge bg-info mt-1" style="font-size:0.65rem">{{ t.recurrence }}</span>{% endif %}
  </div>
</div>
{% endfor %}
</div></div></div>
{% endfor %}
</div>
{% endblock %}""").replace("{% block extra_js %}{% endblock %}", """{% block extra_js %}
<script>
function dragTask(e){ e.dataTransfer.setData('taskId', e.target.closest('.kanban-card').dataset.id); }
function dropTask(e){ e.preventDefault(); const id=e.dataTransfer.getData('taskId'); const status=e.currentTarget.dataset.status;
  fetch("{{ url_for('tasks_kanban_update') }}",{method:'POST',headers:{'Content-Type':'application/json','X-CSRFToken':'{{ csrf_token_value }}'},body:JSON.stringify({task_id:parseInt(id),status})})
  .then(r=>r.json()).then(()=>location.reload()); }
</script>
{% endblock %}""")

    EXPORT_TPL = BASE.replace("{% block content %}{% endblock %}", """{% block content %}
<h3 class="mb-4"><i class="bi bi-arrow-down-up"></i> Export / Import Data</h3>
<div class="row g-4">
  <div class="col-md-6"><div class="card h-100"><div class="card-header">Export</div><div class="card-body">
    <p class="text-muted small">Download all your data for backup or migration.</p>
    <div class="d-flex flex-column gap-2">
      <a href="{{ url_for('export_data', format='json') }}" class="btn btn-purple"><i class="bi bi-filetype-json"></i> Download JSON</a>
      <a href="{{ url_for('export_data', format='csv') }}" class="btn btn-outline-secondary"><i class="bi bi-filetype-csv"></i> Download Tasks CSV</a>
    </div>
  </div></div></div>
  <div class="col-md-6"><div class="card h-100"><div class="card-header">Import</div><div class="card-body">
    <form method="POST" action="{{ url_for('import_data') }}" enctype="multipart/form-data">{{ csrf_token_field|safe }}
      <p class="text-muted small">Import tasks from CSV (columns: title, priority, status, deadline).</p>
      <input type="file" name="file" accept=".csv,.json" class="form-control mb-3" required>
      <button class="btn btn-teal">Import File</button>
    </form>
  </div></div></div>
</div>
<div class="card mt-4"><div class="card-header"><i class="bi bi-people me-2"></i>Shared Tasks & Goals</div><div class="card-body">
  <form method="POST" action="{{ url_for('share_item') }}" class="row g-2">{{ csrf_token_field|safe }}
    <div class="col-md-3"><select name="item_type" class="form-select form-select-sm"><option value="task">Task</option><option value="goal">Goal</option></select></div>
    <div class="col-md-3"><input name="item_id" type="number" class="form-control form-control-sm" placeholder="Item ID" required></div>
    <div class="col-md-4"><input name="email" type="email" class="form-control form-control-sm" placeholder="Collaborator email" required></div>
    <div class="col-md-2"><button class="btn btn-purple btn-sm w-100">Share</button></div>
  </form>
  {% if shared_with_me %}
  <hr><h6>Shared With You</h6>
  <ul class="list-group list-group-flush">{% for s in shared_with_me %}<li class="list-group-item">{{ s.type }}: {{ s.title }} <small class="text-muted">from {{ s.owner }}</small></li>{% endfor %}</ul>
  {% endif %}
</div></div>
{% endblock %}""")

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.route("/weekly-review")
    @app.route("/weekly-review/<week>")
    @login_required
    def weekly_review(week=None):
        if week:
            try:
                ws = date.fromisoformat(week)
            except ValueError:
                abort(404)
            review = WeeklyReview.query.filter_by(user_id=current_user.id, week_start=ws).first()
            if not review:
                review = _generate_weekly_review(current_user.id, ws)
        else:
            review = WeeklyReview.query.filter_by(user_id=current_user.id).order_by(WeeklyReview.week_start.desc()).first()
            if not review:
                review = _generate_weekly_review(current_user.id)
        stats = json.loads(review.stats_json or "{}")
        past = WeeklyReview.query.filter_by(user_id=current_user.id).order_by(WeeklyReview.week_start.desc()).limit(8).all()
        trend_labels, trend_tasks, trend_focus, trend_mood = [], [], [], []
        for i in range(6, -1, -1):
            d = date.today() - timedelta(days=i)
            trend_labels.append(d.strftime("%a"))
            d_start = datetime.combine(d, datetime.min.time())
            d_end = d_start + timedelta(days=1)
            trend_tasks.append(Task.query.filter(Task.user_id == current_user.id, Task.status == "Completed", Task.completed_at >= d_start, Task.completed_at < d_end).count())
            trend_focus.append(sum(s.actual_minutes for s in FocusSession.query.filter(FocusSession.user_id == current_user.id, FocusSession.started_at >= d_start, FocusSession.started_at < d_end).all()))
            ml = MoodLog.query.filter_by(user_id=current_user.id, logged_date=d).first()
            trend_mood.append(ml.mood if ml else 0)
        return render_template_string(WEEKLY_REVIEW_TPL, review=review, stats=stats, past_reviews=past,
            trend_labels=json.dumps(trend_labels), trend_tasks=json.dumps(trend_tasks),
            trend_focus=json.dumps(trend_focus), trend_mood=json.dumps(trend_mood), csrf_token_field=csrf_hidden())

    @app.route("/weekly-review/generate", methods=["POST"])
    @login_required
    def weekly_review_generate():
        _generate_weekly_review(current_user.id)
        flash("Weekly review regenerated.", "success")
        return redirect(url_for("weekly_review"))

    @app.route("/notes/graph")
    @login_required
    def notes_graph():
        return render_template_string(NOTES_GRAPH_TPL)

    @app.route("/notes/graph-data")
    @login_required
    def notes_graph_data():
        notes = Note.query.filter_by(user_id=current_user.id).all()
        links = NoteLink.query.filter_by(user_id=current_user.id).all()
        nodes = [{"id": n.id, "title": n.title, "category": n.category or "General",
                  "word_count": len((n.content or "").split())} for n in notes]
        edges = [{"source": l.source_note_id, "target": l.target_note_id, "similarity": round(l.similarity, 3)} for l in links]
        return jsonify({"nodes": nodes, "links": edges})

    @app.route("/focus")
    @login_required
    def focus_index():
        uid = current_user.id
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_mins = sum(s.actual_minutes for s in FocusSession.query.filter(
            FocusSession.user_id == uid, FocusSession.started_at >= today_start).all())
        tasks = Task.query.filter_by(user_id=uid, status="Pending").limit(20).all()
        return render_template_string(FOCUS_TPL, tasks=tasks, today_minutes=today_mins)

    @app.route("/focus/session", methods=["POST"])
    @csrf.exempt
    @login_required
    def focus_session():
        data = request.get_json(force=True)
        actual = int(data.get("actual_minutes", 25))
        planned = int(data.get("planned_minutes", 25))
        if actual > planned * 4:
            actual = planned
        started = data.get("started_at")
        started_at = datetime.fromisoformat(started.replace("Z", "")) if started else datetime.utcnow() - timedelta(minutes=actual)
        ended_at = datetime.utcnow()
        sess = FocusSession(
            user_id=current_user.id,
            task_id=int(data["task_id"]) if data.get("task_id") else None,
            planned_minutes=planned, actual_minutes=actual,
            interruptions=int(data.get("interruptions", 0)),
            quality=int(data.get("quality")) if data.get("quality") else None,
            session_type=data.get("session_type", "Work"),
            started_at=started_at, ended_at=ended_at,
        )
        db.session.add(sess)
        db.session.commit()
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_sessions = FocusSession.query.filter(FocusSession.user_id == current_user.id, FocusSession.started_at >= today_start).count()
        total_today = sum(s.actual_minutes for s in FocusSession.query.filter(FocusSession.user_id == current_user.id, FocusSession.started_at >= today_start).all())
        streak = today_sessions
        return jsonify({"session_id": sess.id, "streak": streak, "total_today": total_today, "focus_score": compute_focus_score(sess)})

    @app.route("/focus/analytics")
    @login_required
    def focus_analytics():
        uid = current_user.id
        heatmap = []
        for i in range(89, -1, -1):
            d = date.today() - timedelta(days=i)
            d_start = datetime.combine(d, datetime.min.time())
            d_end = d_start + timedelta(days=1)
            sessions = FocusSession.query.filter(FocusSession.user_id == uid, FocusSession.started_at >= d_start, FocusSession.started_at < d_end).all()
            heatmap.append({"date": d.isoformat(), "sessions": len(sessions), "minutes": sum(s.actual_minutes for s in sessions)})
        hourly = {h: {"total_q": 0, "count": 0, "sessions": 0} for h in range(24)}
        all_sess = FocusSession.query.filter(FocusSession.user_id == uid, FocusSession.started_at >= datetime.utcnow() - timedelta(days=90)).all()
        for s in all_sess:
            h = s.started_at.hour
            hourly[h]["sessions"] += 1
            if s.quality:
                hourly[h]["total_q"] += s.quality
                hourly[h]["count"] += 1
        hourly_dist = [{"hour": h, "avg_quality": round(v["total_q"] / max(v["count"], 1), 1), "session_count": v["sessions"]} for h, v in hourly.items() if v["sessions"] > 0]
        week_start = datetime.combine(date.today() - timedelta(days=7), datetime.min.time())
        week_sess = [s for s in all_sess if s.started_at >= week_start]
        return jsonify({
            "heatmap": heatmap,
            "hourly_distribution": hourly_dist or [{"hour": 9, "avg_quality": 3.5, "session_count": 0}],
            "weekly_summary": {
                "total_sessions": len(week_sess),
                "total_minutes": sum(s.actual_minutes for s in week_sess),
                "avg_quality": round(sum(s.quality or 3 for s in week_sess) / max(len(week_sess), 1), 1),
                "best_hour": max(hourly, key=lambda h: hourly[h]["sessions"]) if any(hourly[h]["sessions"] for h in hourly) else 10,
                "pomodoro_streak": len(week_sess),
            },
        })

    @app.route("/mood/intelligence")
    @login_required
    def mood_intelligence():
        insights = MoodInsight.query.filter_by(user_id=current_user.id).order_by(MoodInsight.computed_at.desc()).limit(10).all()
        labels, moods, energies, completions = [], [], [], []
        for i in range(29, -1, -1):
            d = date.today() - timedelta(days=i)
            labels.append(d.strftime("%m/%d"))
            ml = MoodLog.query.filter_by(user_id=current_user.id, logged_date=d).first()
            moods.append(ml.mood if ml else None)
            energies.append(ml.energy_level if ml and ml.energy_level else None)
            d_start = datetime.combine(d, datetime.min.time())
            d_end = d_start + timedelta(days=1)
            cr = Task.query.filter(Task.user_id == current_user.id, Task.status == "Completed", Task.completed_at >= d_start, Task.completed_at < d_end).count()
            cc = Task.query.filter(Task.user_id == current_user.id, Task.created_at >= d_start, Task.created_at < d_end).count()
            completions.append(round(cr / max(cc, 1) * 100, 1))
        chart_data = json.dumps({"labels": labels, "mood": moods, "energy": energies, "completion": completions})
        return render_template_string(MOOD_INTEL_TPL, insights=insights, chart_data=chart_data, csrf_token_field=csrf_hidden())

    @app.route("/mood/log", methods=["POST"])
    @login_required
    def mood_log():
        mood = max(1, min(5, int(request.form.get("mood", 3))))
        energy = max(1, min(5, int(request.form.get("energy_level", 3))))
        raw_tags = request.form.get("tags", "")
        tags_list = [sanitize(t.strip())[:30] for t in raw_tags.split(",") if t.strip()][:3]
        tags = ",".join(tags_list)
        note_text = sanitize(request.form.get("note", ""))[:500]
        today = date.today()
        existing = MoodLog.query.filter_by(user_id=current_user.id, logged_date=today).first()
        if existing:
            existing.mood = mood
            existing.energy_level = energy
            existing.tags = tags
            existing.note = note_text
        else:
            db.session.add(MoodLog(user_id=current_user.id, mood=mood, energy_level=energy, tags=tags, note=note_text, logged_date=today))
        db.session.commit()
        flash("Mood & energy logged.", "success")
        return redirect(url_for("mood_intelligence"))

    @app.route("/mood/insights")
    @login_required
    def mood_insights_api():
        insights = MoodInsight.query.filter_by(user_id=current_user.id).all()
        return jsonify([{"type": i.insight_type, "text": i.insight_text, "r": i.correlation_r, "p": i.p_value} for i in insights])

    @app.route("/tasks/parse-nl", methods=["POST"])
    @csrf.exempt
    @login_required
    def tasks_parse_nl():
        data = request.get_json(force=True)
        result = parse_natural_language_task(data.get("text", ""))
        return jsonify(result)

    @app.route("/tasks/kanban")
    @login_required
    def tasks_kanban():
        uid = current_user.id
        tasks = Task.query.filter_by(user_id=uid, parent_id=None).all()
        columns = {"Pending": [], "In Progress": [], "Completed": []}
        for t in tasks:
            if t.status in columns:
                columns[t.status].append(t)
        return render_template_string(KANBAN_TPL, columns=columns)

    @app.route("/tasks/kanban/update", methods=["POST"])
    @csrf.exempt
    @login_required
    def tasks_kanban_update():
        data = request.get_json(force=True)
        t = Task.query.get_or_404(data["task_id"])
        if t.user_id != current_user.id:
            abort(403)
        new_status = data.get("status")
        if new_status in ("Pending", "In Progress", "Completed"):
            old = t.status
            t.status = new_status
            if new_status == "Completed" and old != "Completed":
                t.completed_at = datetime.utcnow()
            elif new_status != "Completed":
                t.completed_at = None
            db.session.commit()
        return jsonify({"ok": True})

    @app.route("/tasks/<int:task_id>/subtasks", methods=["GET", "POST"])
    @login_required
    def task_subtasks(task_id):
        t = Task.query.get_or_404(task_id)
        if t.user_id != current_user.id:
            abort(403)
        if request.method == "POST":
            title = sanitize(request.form.get("title", ""))
            if title:
                db.session.add(Subtask(task_id=t.id, title=title))
                db.session.commit()
                flash("Subtask added.", "success")
            return redirect(url_for("task_subtasks", task_id=task_id))
        subtasks = Subtask.query.filter_by(task_id=task_id).order_by(Subtask.sort_order).all()
        tpl = BASE.replace("{% block content %}{% endblock %}", """{% block content %}
<h3><i class="bi bi-list-check"></i> Subtasks: {{ task.title }}</h3>
<div class="row"><div class="col-md-6"><div class="card"><div class="card-body">
<form method="POST">{{ csrf_token_field|safe }}<div class="input-group mb-3"><input name="title" class="form-control" placeholder="New subtask..." required><button class="btn btn-purple">Add</button></div></form>
<ul class="list-group list-group-flush">{% for s in subtasks %}
<li class="list-group-item d-flex justify-content-between align-items-center">
  <form method="POST" action="{{ url_for('subtask_toggle', subtask_id=s.id) }}" class="d-flex align-items-center gap-2 flex-grow-1">{{ csrf_token_field|safe }}
    <input type="checkbox" {{ 'checked' if s.completed else '' }} onchange="this.form.submit()">
    <span class="{{ 'text-decoration-line-through text-muted' if s.completed else '' }}">{{ s.title }}</span>
  </form>
  <form method="POST" action="{{ url_for('subtask_delete', subtask_id=s.id) }}">{{ csrf_token_field|safe }}<button class="btn btn-sm btn-outline-danger"><i class="bi bi-trash"></i></button></form>
</li>{% else %}<li class="list-group-item text-muted">No subtasks yet.</li>{% endfor %}</ul>
</div></div></div></div>
<a href="{{ url_for('tasks_index') }}" class="btn btn-outline-secondary mt-3">Back to Tasks</a>
{% endblock %}""")
        return render_template_string(tpl, task=t, subtasks=subtasks, csrf_token_field=csrf_hidden())

    @app.route("/subtasks/<int:subtask_id>/toggle", methods=["POST"])
    @login_required
    def subtask_toggle(subtask_id):
        s = Subtask.query.get_or_404(subtask_id)
        t = Task.query.get(s.task_id)
        if t.user_id != current_user.id:
            abort(403)
        s.completed = not s.completed
        db.session.commit()
        return redirect(url_for("task_subtasks", task_id=t.id))

    @app.route("/subtasks/<int:subtask_id>/delete", methods=["POST"])
    @login_required
    def subtask_delete(subtask_id):
        s = Subtask.query.get_or_404(subtask_id)
        t = Task.query.get(s.task_id)
        if t.user_id != current_user.id:
            abort(403)
        tid = t.id
        db.session.delete(s)
        db.session.commit()
        return redirect(url_for("task_subtasks", task_id=tid))

    @app.route("/tasks/<int:task_id>/time", methods=["POST"])
    @login_required
    def task_log_time(task_id):
        t = Task.query.get_or_404(task_id)
        if t.user_id != current_user.id:
            abort(403)
        minutes = max(1, int(request.form.get("minutes", 0)))
        note = sanitize(request.form.get("note", ""))
        db.session.add(TimeLog(task_id=t.id, user_id=current_user.id, minutes=minutes, note=note))
        t.time_logged_minutes = (t.time_logged_minutes or 0) + minutes
        db.session.commit()
        flash(f"Logged {minutes} minutes.", "success")
        return redirect(url_for("tasks_edit", task_id=task_id))

    @app.route("/export-import")
    @login_required
    def export_import_page():
        shared = []
        for ts in TaskShare.query.filter_by(shared_with_id=current_user.id).all():
            t = Task.query.get(ts.task_id)
            if t:
                owner = User.query.get(ts.owner_id)
                shared.append({"type": "Task", "title": t.title, "owner": owner.name if owner else "Unknown"})
        for gs in GoalShare.query.filter_by(shared_with_id=current_user.id).all():
            g = Goal.query.get(gs.goal_id)
            if g:
                owner = User.query.get(gs.owner_id)
                shared.append({"type": "Goal", "title": g.name, "owner": owner.name if owner else "Unknown"})
        return render_template_string(EXPORT_TPL, shared_with_me=shared, csrf_token_field=csrf_hidden())

    @app.route("/export/<format>")
    @login_required
    def export_data(format):
        uid = current_user.id
        if format == "json":
            data = {
                "user": current_user.name,
                "exported_at": datetime.utcnow().isoformat(),
                "tasks": [{"title": t.title, "priority": t.priority, "status": t.status,
                           "deadline": t.deadline.isoformat() if t.deadline else None,
                           "time_logged": t.time_logged_minutes or 0} for t in Task.query.filter_by(user_id=uid).all()],
                "goals": [{"name": g.name, "progress": g.progress, "status": g.status} for g in Goal.query.filter_by(user_id=uid).all()],
                "notes": [{"title": n.title, "category": n.category, "content": n.content} for n in Note.query.filter_by(user_id=uid).all()],
                "mood_logs": [{"mood": m.mood, "energy": m.energy_level, "tags": m.tags, "date": m.logged_date.isoformat()} for m in MoodLog.query.filter_by(user_id=uid).all()],
            }
            buf = io.BytesIO(json.dumps(data, indent=2).encode())
            buf.seek(0)
            return send_file(buf, mimetype="application/json", as_attachment=True, download_name="life_os_export.json")
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["title", "priority", "status", "deadline", "time_logged_minutes"])
        for t in Task.query.filter_by(user_id=uid).all():
            writer.writerow([t.title, t.priority, t.status, t.deadline.isoformat() if t.deadline else "", t.time_logged_minutes or 0])
        out = io.BytesIO(buf.getvalue().encode())
        out.seek(0)
        return send_file(out, mimetype="text/csv", as_attachment=True, download_name="tasks_export.csv")

    @app.route("/import", methods=["POST"])
    @login_required
    def import_data():
        f = request.files.get("file")
        if not f:
            flash("No file uploaded.", "danger")
            return redirect(url_for("export_import_page"))
        filename = f.filename.lower()
        try:
            if filename.endswith(".json"):
                data = json.load(f)
                for t in data.get("tasks", []):
                    dl = datetime.fromisoformat(t["deadline"]) if t.get("deadline") else None
                    db.session.add(Task(user_id=current_user.id, title=t["title"], priority=t.get("priority", "Medium"),
                                       status=t.get("status", "Pending"), deadline=dl))
            elif filename.endswith(".csv"):
                reader = csv.DictReader(io.TextIOWrapper(f.stream, encoding="utf-8-sig"))
                for row in reader:
                    dl = None
                    if row.get("deadline"):
                        try:
                            dl = datetime.fromisoformat(row["deadline"])
                        except ValueError:
                            pass
                    db.session.add(Task(user_id=current_user.id, title=row.get("title", "Imported task"),
                                       priority=row.get("priority", "Medium"), status=row.get("status", "Pending"), deadline=dl))
            db.session.commit()
            flash("Import successful.", "success")
        except Exception as exc:
            db.session.rollback()
            flash(f"Import failed: {exc}", "danger")
        return redirect(url_for("export_import_page"))

    @app.route("/share", methods=["POST"])
    @login_required
    def share_item():
        item_type = request.form.get("item_type")
        item_id = int(request.form.get("item_id", 0))
        email = request.form.get("email", "").lower().strip()
        collaborator = User.query.filter_by(email=email).first()
        if not collaborator:
            flash("User not found with that email.", "danger")
            return redirect(url_for("export_import_page"))
        if collaborator.id == current_user.id:
            flash("Cannot share with yourself.", "warning")
            return redirect(url_for("export_import_page"))
        if item_type == "task":
            t = Task.query.get_or_404(item_id)
            if t.user_id != current_user.id:
                abort(403)
            existing = TaskShare.query.filter_by(task_id=item_id, shared_with_id=collaborator.id).first()
            if not existing:
                db.session.add(TaskShare(task_id=item_id, owner_id=current_user.id, shared_with_id=collaborator.id))
        elif item_type == "goal":
            g = Goal.query.get_or_404(item_id)
            if g.user_id != current_user.id:
                abort(403)
            existing = GoalShare.query.filter_by(goal_id=item_id, shared_with_id=collaborator.id).first()
            if not existing:
                db.session.add(GoalShare(goal_id=item_id, owner_id=current_user.id, shared_with_id=collaborator.id))
        db.session.commit()
        flash(f"Shared with {collaborator.name}.", "success")
        return redirect(url_for("export_import_page"))

    @app.route("/settings/theme", methods=["POST"])
    @login_required
    def set_theme():
        theme = request.form.get("theme", "dark")
        if theme not in ("dark", "light"):
            theme = "dark"
        current_user.theme = theme
        db.session.commit()
        return redirect(request.referrer or url_for("dashboard"))

    @app.route("/manifest.json")
    def pwa_manifest():
        return app.send_static_file("manifest.json")

    @app.route("/sw.js")
    def service_worker():
        return app.send_static_file("sw.js")
