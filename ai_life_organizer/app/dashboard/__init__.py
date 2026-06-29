from flask import Blueprint

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/", template_folder="templates")

from app.dashboard import routes  # noqa: E402, F401
