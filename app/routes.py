from flask import Blueprint, render_template, jsonify
from app.utils import get_all_reviews  # Pastikan get_all_reviews sudah dibuat di app/utils.py

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/preprocessing')
def preprocessing_page():
    # Mengambil data ulasan beserta seluruh kolom step1 - step5 dari MySQL
    reviews_data = get_all_reviews()
    return render_template('preprocessing.html', reviews=reviews_data)

@main_bp.route('/api/health')
def health_check():
    return jsonify({"status": "ok", "message": "Server Flask berjalan lancar!"})