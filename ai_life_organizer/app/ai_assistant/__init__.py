from flask import Blueprint

ai_bp = Blueprint("ai", __name__, url_prefix="/ai", template_folder="templates")

from app.ai_assistant import routes  # noqa: E402, F401
