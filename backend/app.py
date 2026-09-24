import os

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from backend.routes.chat_routes import chat_bp
from backend.config import Config
from backend.extensions import db, jwt
from backend.routes.abs_routes import abs_bp
from backend.routes.source_routes import source_bp
from backend.routes.tkdl_routes import tkdl_bp
from backend.routes.classification_routes import classification_bp
from backend.routes.feedback_routes import feedback_bp
from backend.routes.auth_routes import auth_bp
from backend.routes.history_routes import history_bp


# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)
app.config.from_object(Config)


# ============================================================
# Project / Frontend Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)


# ============================================================
# CORS
# ============================================================

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "*"
        }
    }
)


# ============================================================
# Database & JWT
# ============================================================

db.init_app(app)
jwt.init_app(app)


# ============================================================
# SQLAlchemy Models
# ============================================================

# Import models so SQLAlchemy knows about them.
# IMPORTANT:
# db.create_all() is intentionally NOT called here.
# This prevents database initialization during Gunicorn startup.
from backend import models  # noqa: F401


# ============================================================
# API Blueprints
# ============================================================

app.register_blueprint(
    chat_bp,
    url_prefix="/api"
)

app.register_blueprint(
    abs_bp,
    url_prefix="/api"
)

app.register_blueprint(
    source_bp,
    url_prefix="/api"
)

app.register_blueprint(
    tkdl_bp,
    url_prefix="/api"
)

app.register_blueprint(
    classification_bp,
    url_prefix="/api"
)

app.register_blueprint(
    feedback_bp,
    url_prefix="/api"
)

app.register_blueprint(
    auth_bp,
    url_prefix="/api"
)

app.register_blueprint(
    history_bp,
    url_prefix="/api"
)


# ============================================================
# JWT Error Handlers
# ============================================================

@jwt.unauthorized_loader
def _unauthorized(reason):

    return jsonify({
        "success": False,
        "error": "Authentication required. Please log in."
    }), 401


@jwt.invalid_token_loader
def _invalid_token(reason):

    return jsonify({
        "success": False,
        "error": "Invalid or expired session. Please log in again."
    }), 401


@jwt.expired_token_loader
def _expired_token(jwt_header, jwt_payload):

    return jsonify({
        "success": False,
        "error": "Your session has expired. Please log in again."
    }), 401


# ============================================================
# FRONTEND
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/<path:path>", methods=["GET"])
def serve_frontend(path):

    file_path = os.path.join(
        FRONTEND_DIR,
        path
    )

    # Serve existing CSS, JS, images, pages, etc.
    if os.path.isfile(file_path):

        return send_from_directory(
            FRONTEND_DIR,
            path
        )

    # For frontend routes, return index.html
    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# Backend Health Check
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "service": "IP-SAKTI Sahayak"
    })


# ============================================================
# Global Error Handler
# ============================================================

@app.errorhandler(Exception)
def handle_error(error):

    print("Server Error:", error)

    return jsonify({
        "error": "Internal server error"
    }), 500


# ============================================================
# Local Development
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=True
    )