# Implemented in Task 16
from flask import render_template
from flask_login import login_required
from app.dashboard import dashboard_bp


@dashboard_bp.route("/")
@login_required
def index():
    return render_template("dashboard/index.html")
