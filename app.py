from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os

# --- 1. INITIALIZATION (Must come first!) ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- 2. DATABASE MODEL ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), nullable=False) 

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- 3. ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
            
    return render_template('login.html')

# --- THIS IS THE NEW REGISTER ROUTE ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role') 
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
        
        new_user = User(username=username, role=role)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')
# --------------------------------------

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction_text = ""
    risk_level = ""
    
    if request.method == 'POST':
        try:
            # Simple dummy logic for now
            study_time = float(request.form.get('study_time'))
            failures = int(request.form.get('failures'))
            absences = int(request.form.get('absences'))
            
            if failures > 0 or absences > 10:
                risk_level = "High Risk"
                prediction_text = "Student is likely to struggle academically."
            elif study_time < 2:
                risk_level = "Moderate Risk"
                prediction_text = "Student may need monitoring."
            else:
                risk_level = "Low Risk"
                prediction_text = "Student is on track for success."
                
        except ValueError:
            prediction_text = "Error: Please enter valid numbers."

    return render_template('dashboard.html', 
                           name=current_user.username,
                           role=current_user.role,
                           prediction=prediction_text,
                           risk=risk_level)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create test admin if missing
        if not User.query.filter_by(username='admin').first():
            print("Creating test user: admin")
            admin = User(username='admin', role='admin')
            admin.set_password('password123')
            db.session.add(admin)
            db.session.commit()
            
    app.run(debug=True)