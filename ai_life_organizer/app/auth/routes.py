from flask import render_template, redirect, url_for, flash, request, abort
from flask_login import login_user, logout_user, login_required, current_user
from app import db, mail
from app.auth import auth_bp
from app.auth.forms import RegistrationForm, LoginForm, PasswordResetRequestForm, PasswordResetForm, ProfileForm
from app.models import User, PasswordResetToken
from app.utils.security import sanitize
from datetime import datetime, timedelta
import hashlib
import secrets


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(name=sanitize(form.name.data), email=form.email.data.lower())
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Account created. Welcome!", "success")
        return redirect(url_for("dashboard.index"))
    return render_template("auth/register.html", form=form)


# ---------------------------------------------------------------------------
# Login / Logout
# ---------------------------------------------------------------------------
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get("next")
            flash("Logged in.", "success")
            return redirect(next_page or url_for("dashboard.index"))
        flash("Invalid email or password.", "danger")
    return render_template("auth/login.html", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


# ---------------------------------------------------------------------------
# Password Reset
# ---------------------------------------------------------------------------
def _generate_reset_token(user):
    raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    expires_at = datetime.utcnow() + timedelta(minutes=60)
    # Invalidate existing unused tokens
    PasswordResetToken.query.filter_by(user_id=user.id, used=False).update({"used": True})
    token_record = PasswordResetToken(user_id=user.id, token_hash=token_hash, expires_at=expires_at)
    db.session.add(token_record)
    db.session.commit()
    return raw


@auth_bp.route("/reset-request", methods=["GET", "POST"])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    form = PasswordResetRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        if user:
            raw_token = _generate_reset_token(user)
            reset_url = url_for("auth.reset_password", token=raw_token, _external=True)
            try:
                from flask_mail import Message
                msg = Message("Password Reset Request", recipients=[user.email])
                msg.body = f"Click the link to reset your password: {reset_url}\nThis link expires in 60 minutes."
                mail.send(msg)
            except Exception:
                pass  # Don't expose whether mail succeeded to avoid user enumeration
        # Always show same message to prevent user enumeration
        flash("If that email is registered, a reset link has been sent.", "info")
        return redirect(url_for("auth.login"))
    return render_template("auth/reset_request.html", form=form)


@auth_bp.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    token_record = PasswordResetToken.query.filter_by(token_hash=token_hash, used=False).first()
    if not token_record or token_record.expires_at < datetime.utcnow():
        flash("The reset link is invalid or has expired.", "danger")
        return redirect(url_for("auth.reset_request"))
    form = PasswordResetForm()
    if form.validate_on_submit():
        user = User.query.get(token_record.user_id)
        user.set_password(form.password.data)
        token_record.used = True
        db.session.commit()
        flash("Password updated. Please log in.", "success")
        return redirect(url_for("auth.login"))
    return render_template("auth/reset_password.html", form=form)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    form = ProfileForm(obj=current_user)
    if form.validate_on_submit():
        new_email = form.email.data.lower()
        if new_email != current_user.email:
            existing = User.query.filter_by(email=new_email).first()
            if existing:
                flash("Email is already in use by another account.", "danger")
                return render_template("auth/profile.html", form=form)
        current_user.name = sanitize(form.name.data)
        current_user.email = new_email
        db.session.commit()
        flash("Profile updated.", "success")
        return redirect(url_for("auth.profile"))
    return render_template("auth/profile.html", form=form)


@auth_bp.route("/delete-account", methods=["POST"])
@login_required
def delete_account():
    user = current_user._get_current_object()
    logout_user()
    db.session.delete(user)
    db.session.commit()
    flash("Your account and all data have been permanently deleted.", "info")
    return redirect(url_for("auth.register"))
