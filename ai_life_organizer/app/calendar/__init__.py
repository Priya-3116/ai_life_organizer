from flask import Blueprint

calendar_bp = Blueprint("calendar", __name__, url_prefix="/calendar", template_folder="templates")

from app.calendar import routes  # noqa: E402, F401
