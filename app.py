from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from flask_socketio import SocketIO, emit
from datetime import datetime
import random
import math
import threading
import time
# ml model imports below 
import numpy as np
import joblib
import tensorflow as tf

# ─────────────────────────────────────────────────────────────
# ML MODEL SETUP
# ─────────────────────────────────────────────────────────────

scaler = joblib.load('scaler.pkl')

interpreter = tf.lite.Interpreter(model_path='fall_model.tflite')
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Buffer to collect 256 timesteps per device before running prediction
sensor_buffers = {}  # device_id -> list of [ax,ay,az,gx,gy,gz] readings
last_fall_time = {}  # device_id -> timestamp of last fall detection
FALL_COOLDOWN_SECONDS = 10

def run_fall_prediction(device_id):
    buf = sensor_buffers.get(device_id, [])
    if len(buf) < 50:
        return False, 0.0

    # Always feed 256 timesteps — pad with zeros if buffer not full yet
    if len(buf) >= 256:
        window = np.array(buf[-256:], dtype=np.float32)
    else:
        window = np.array(buf, dtype=np.float32)
        pad = np.zeros((256 - len(window), 6), dtype=np.float32)
        window = np.vstack([pad, window])

    window_scaled = scaler.transform(window).reshape(1, 256, 6)

    interpreter.set_tensor(input_details[0]['index'], window_scaled)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_class = int(np.argmax(output))
    confidence = float(np.max(output)) * 100

    fall_detected = predicted_class in (1, 2)
    return fall_detected, confidence

app = Flask(__name__)
app.config['SECRET_KEY'] = 'guardiansense-secret-key-change-in-prod'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///guardiansense.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db      = SQLAlchemy(app)
bcrypt  = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False)
# ─────────────────────────────────────────────────────────────
# TWILIO SMS + CALL SETUP
# ─────────────────────────────────────────────────────────────
from twilio.rest import Client as TwilioClient

TWILIO_SID   = 'AC323c02ebe52a9d74356caf2850af5e14'
TWILIO_TOKEN = 'febfb6b3c258686fe932ef1981a49160'
TWILIO_FROM  = '+19894743968'  # your Twilio number

twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)

def send_fall_sms(to_number, elder_name, confidence, lat, lng):
    if not to_number:
        return
    try:
        maps_link = f"https://maps.google.com/?q={lat},{lng}" if lat and lng else "Location unavailable"
        body = (
            f"🚨 FALL ALERT — GuardianSense\n"
            f"Elder: {elder_name}\n"
            f"Confidence: {confidence:.1f}%\n"
            f"Location: {maps_link}\n"
            f"Please respond immediately."
        )
        twilio_client.messages.create(
            body=body,
            from_=TWILIO_FROM,
            to=to_number
        )
        print(f"SMS sent to {to_number}")
    except Exception as e:
        print(f"SMS failed: {e}")

def send_fall_call(to_number, elder_name):
    if not to_number:
        return
    try:
        twilio_client.calls.create(
            twiml=f'<Response><Say voice="alice">Emergency alert. {elder_name} has fallen. Please respond immediately.</Say><Pause length="1"/><Say voice="alice">This is an automated alert from Guardian Sense.</Say></Response>',
            from_=TWILIO_FROM,
            to=to_number
        )
        print(f"Call initiated to {to_number}")
    except Exception as e:
        print(f"Call failed: {e}")
# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(120), nullable=False)
    email        = db.Column(db.String(120), unique=True, nullable=False)
    password     = db.Column(db.String(200), nullable=False)
    role         = db.Column(db.String(20), nullable=False)   # admin | caretaker | elder
    device_id    = db.Column(db.String(64), nullable=True) # only for elders
    phone        = db.Column(db.String(20), nullable=True)
    caretaker_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # elder → caretaker
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    elder_charges = db.relationship('User', backref=db.backref('caretaker', remote_side=[id]), lazy='dynamic')
    sensor_logs   = db.relationship('SensorLog', backref='elder', lazy='dynamic', foreign_keys='SensorLog.elder_id')
    fall_events   = db.relationship('FallEvent', backref='elder', lazy='dynamic', foreign_keys='FallEvent.elder_id')

class SensorLog(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    elder_id  = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ax        = db.Column(db.Float)
    ay        = db.Column(db.Float)
    az        = db.Column(db.Float)
    gx        = db.Column(db.Float)
    gy        = db.Column(db.Float)
    gz        = db.Column(db.Float)
    accel_mag = db.Column(db.Float)
    lat       = db.Column(db.Float, nullable=True)
    lng       = db.Column(db.Float, nullable=True)
    gps_fixed = db.Column(db.Boolean, default=False)
    battery   = db.Column(db.Integer, default=100)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class FallEvent(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    elder_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    confidence   = db.Column(db.Float)
    lat          = db.Column(db.Float, nullable=True)
    lng          = db.Column(db.Float, nullable=True)
    acknowledged = db.Column(db.Boolean, default=False)
    ack_by       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    ack_at       = db.Column(db.DateTime, nullable=True)
    timestamp    = db.Column(db.DateTime, default=datetime.utcnow)

class Notification(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message    = db.Column(db.String(500))
    type       = db.Column(db.String(30), default='info')   # info | warning | danger
    read       = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for(f"{current_user.role}_dashboard"))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        if User.query.filter_by(email=data['email']).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        hashed = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        new_user = User(
            name=data['name'],
            email=data['email'],
            password=hashed,
            role=data['role'],
            device_id=data.get('device_id') if data['role'] == 'elder' else None
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    caretakers = User.query.filter_by(role='caretaker').all()
    return render_template('register.html', caretakers=caretakers)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for(f"{current_user.role}_dashboard"))
    if request.method == 'POST':
        email    = request.form.get('email')
        password = request.form.get('password')
        user     = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for(f"{user.role}_dashboard"))
        flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ─────────────────────────────────────────────────────────────
# DASHBOARD ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        return redirect(url_for('home'))
    elders     = User.query.filter_by(role='elder').all()
    caretakers = User.query.filter_by(role='caretaker').all()
    total_falls = FallEvent.query.count()
    pending     = FallEvent.query.filter_by(acknowledged=False).count()
    return render_template('admin.html',
        elders=elders, caretakers=caretakers,
        total_falls=total_falls, pending=pending)

@app.route('/caretaker')
@login_required
def caretaker_dashboard():
    if current_user.role != 'caretaker':
        return redirect(url_for('home'))
    elders = User.query.filter_by(caretaker_id=current_user.id).all()
    elder_ids = [e.id for e in elders]
    recent_falls = FallEvent.query.filter(
        FallEvent.elder_id.in_(elder_ids)
    ).order_by(FallEvent.timestamp.desc()).limit(20).all()
    notifications = Notification.query.filter_by(
        user_id=current_user.id, read=False
    ).order_by(Notification.created_at.desc()).all()
    return render_template('caretaker.html',
        elders=elders, recent_falls=recent_falls,
        notifications=notifications)

@app.route('/elder')
@login_required
def elder_dashboard():
    if current_user.role != 'elder':
        return redirect(url_for('home'))
    notifications = Notification.query.filter_by(
        user_id=current_user.id
    ).order_by(Notification.created_at.desc()).limit(20).all()
    my_falls = FallEvent.query.filter_by(
        elder_id=current_user.id
    ).order_by(FallEvent.timestamp.desc()).limit(10).all()
    return render_template('elder.html',
        notifications=notifications, my_falls=my_falls)

# ─────────────────────────────────────────────────────────────
# ADMIN API
# ─────────────────────────────────────────────────────────────

@app.route('/api/admin/users', methods=['GET'])
@login_required
def admin_get_users():
    if current_user.role != 'admin':
        return jsonify({'error': 'Forbidden'}), 403
    users = User.query.all()
    return jsonify([{
        'id': u.id, 'name': u.name,
        'email': u.email, 'role': u.role,
        'device_id': u.device_id,
        'created_at': u.created_at.isoformat()
    } for u in users])

@app.route('/api/admin/users', methods=['POST'])
@login_required
def admin_create_user():
    if current_user.role != 'admin':
        return jsonify({'error': 'Forbidden'}), 403
    data = request.get_json()

    # Prevent creating additional admins via the API
    if data.get('role') == 'admin':
        return jsonify({'error': 'Additional admin accounts cannot be created'}), 403

    # Check for duplicate email
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 409

    hashed = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_user = User(
        name=data['name'],
        email=data['email'],
        password=hashed,
        role=data['role'],
        device_id=data.get('device_id'),
        caretaker_id=int(data['caretaker_id']) if data.get('caretaker_id') else None
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'status': 'created', 'id': new_user.id}), 201

@app.route('/api/admin/users/<int:uid>', methods=['DELETE'])
@login_required
def admin_delete_user(uid):
    if current_user.role != 'admin':
        return jsonify({'error': 'Forbidden'}), 403
    user = User.query.get_or_404(uid)
    db.session.delete(user)
    db.session.commit()
    return jsonify({'status': 'deleted'})

@app.route('/api/admin/stats', methods=['GET'])
@login_required
def admin_stats():
    if current_user.role != 'admin':
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify({
        'total_elders':     User.query.filter_by(role='elder').count(),
        'total_caretakers': User.query.filter_by(role='caretaker').count(),
        'total_falls':      FallEvent.query.count(),
        'pending_falls':    FallEvent.query.filter_by(acknowledged=False).count(),
        'total_logs':       SensorLog.query.count(),
    })

# Admin assigns a caretaker to an elder
@app.route('/api/admin/assign', methods=['POST'])
@login_required
def admin_assign():
    if current_user.role != 'admin':
        return jsonify({'error': 'Forbidden'}), 403
    data = request.get_json()
    elder = User.query.get_or_404(data['elder_id'])
    elder.caretaker_id = data['caretaker_id']
    db.session.commit()
    return jsonify({'status': 'assigned'})

# ─────────────────────────────────────────────────────────────
# CARETAKER API
# ─────────────────────────────────────────────────────────────

@app.route('/api/caretaker/elder/<int:elder_id>/sensor')
@login_required
def caretaker_elder_sensor(elder_id):
    if current_user.role not in ('caretaker', 'admin'):
        return jsonify({'error': 'Forbidden'}), 403
    logs = SensorLog.query.filter_by(elder_id=elder_id)\
               .order_by(SensorLog.timestamp.desc()).limit(60).all()
    return jsonify([{
        'ax': l.ax, 'ay': l.ay, 'az': l.az,
        'gx': l.gx, 'gy': l.gy, 'gz': l.gz,
        'accel_mag': l.accel_mag,
        'lat': l.lat, 'lng': l.lng,
        'gps_fixed': l.gps_fixed,
        'battery': l.battery,
        'timestamp': l.timestamp.isoformat()
    } for l in reversed(logs)])

@app.route('/api/caretaker/falls/<int:event_id>/acknowledge', methods=['POST'])
@login_required
def acknowledge_fall(event_id):
    if current_user.role not in ('caretaker', 'admin'):
        return jsonify({'error': 'Forbidden'}), 403
    event = FallEvent.query.get_or_404(event_id)
    event.acknowledged = True
    event.ack_by       = current_user.id
    event.ack_at       = datetime.utcnow()
    db.session.commit()
    socketio.emit('event_acknowledged', {'id': event_id})
    return jsonify({'status': 'acknowledged'})

# ─────────────────────────────────────────────────────────────
# ELDER API
# ─────────────────────────────────────────────────────────────

@app.route('/api/elder/notifications/read', methods=['POST'])
@login_required
def mark_notifications_read():
    Notification.query.filter_by(user_id=current_user.id, read=False)\
        .update({'read': True})
    db.session.commit()
    return jsonify({'status': 'ok'})

# ─────────────────────────────────────────────────────────────
# ESP32 INGEST ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.route('/api/sensor-data', methods=['POST'])
def receive_sensor_data():
    data = request.get_json()
    device_id = data.get('device_id')

    elder = User.query.filter_by(device_id=device_id, role='elder').first()
    if not elder:
        return jsonify({'error': 'Unknown device'}), 404

    ax = data.get('ax', 0)
    ay = data.get('ay', 0)
    az = data.get('az', 0)
    gx = data.get('gx', 0)
    gy = data.get('gy', 0)
    gz = data.get('gz', 0)

    accel_mag = math.sqrt(ax**2 + ay**2 + az**2)

    # Buffer
    if device_id not in sensor_buffers:
        sensor_buffers[device_id] = []

    sensor_buffers[device_id].append([ax, ay, az, gx, gy, gz])

    if len(sensor_buffers[device_id]) > 512:
        sensor_buffers[device_id] = sensor_buffers[device_id][-256:]

    # ✅ FIXED INDENTATION
    fw_fall  = data.get('fw_fall', False)
    gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)

    # Method 1 — Firmware flag (fastest, fires from ESP32 directly)
    if fw_fall:
        fall_detected = True
        confidence    = 78.0

    # Method 2 — Rule based on Flask side (backup)
    elif accel_mag > 25.0 and gyro_mag > 300.0:
        fall_detected = True
        confidence    = 75.0

    # Method 3 — ML model (catches subtle falls firmware misses)
    else:
        fall_detected, confidence = run_fall_prediction(device_id)

    # Cooldown — prevent repeated alerts within 10 seconds
    if fall_detected:
        now_time  = datetime.utcnow()
        last_time = last_fall_time.get(device_id)
        if last_time and (now_time - last_time).total_seconds() < FALL_COOLDOWN_SECONDS:
            fall_detected = False
        else:
            last_fall_time[device_id] = now_time

    # Save log
    log = SensorLog(
        elder_id=elder.id,
        ax=ax, ay=ay, az=az,
        gx=gx, gy=gy, gz=gz,
        accel_mag=accel_mag,
        lat=data.get('lat'), lng=data.get('lng'),
        gps_fixed=data.get('gps_fixed', False),
        battery=data.get('battery', 100)
    )
    db.session.add(log)

    if fall_detected:
        fall = FallEvent(
            elder_id=elder.id,
            confidence=confidence,
            lat=data.get('lat'), lng=data.get('lng')
        )
        db.session.add(fall)
        db.session.flush()

        if elder.caretaker_id:
            db.session.add(Notification(
                user_id=elder.caretaker_id,
                message=f"🚨 Fall detected for {elder.name}! Confidence: {confidence:.1f}%",
                type='danger'
            ))
            # Send SMS and call to caretaker
            caretaker = User.query.get(elder.caretaker_id)
            if caretaker and caretaker.phone:
                threading.Thread(
                    target=send_fall_sms,
                    args=(caretaker.phone, elder.name, confidence, data.get('lat'), data.get('lng')),
                    daemon=True
                ).start()
                threading.Thread(
                    target=send_fall_call,
                    args=(caretaker.phone, elder.name),
                    daemon=True
                ).start()

        db.session.add(Notification(
            user_id=elder.id,
            message="⚠️ A fall event was detected and your caretaker has been alerted.",
            type='warning'
        ))

        db.session.commit()

        socketio.emit('fall_alert', {
            'elder_id': elder.id,
            'elder_name': elder.name,
            'id': fall.id,
            'confidence': confidence,
            'lat': data.get('lat'),
            'lng': data.get('lng'),
            'timestamp': fall.timestamp.isoformat()
        })
    else:
        db.session.commit()

    socketio.emit(f'sensor_{elder.id}', {
        'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz,
        'accel_mag': accel_mag,
        'lat': data.get('lat'), 'lng': data.get('lng'),
        'gps_fixed': data.get('gps_fixed', False),
        'battery': data.get('battery', 100),
        'fall_detected': fall_detected,
        'confidence': confidence,
        'timestamp': datetime.utcnow().isoformat()
    })

    return jsonify({'status': 'ok', 'fall_detected': fall_detected}), 200

# @app.route('/api/sensor-data', methods=['POST'])
# def receive_sensor_data():
#     data      = request.get_json()
#     device_id = data.get('device_id')
#     elder     = User.query.filter_by(device_id=device_id, role='elder').first()
#     if not elder:
#         return jsonify({'error': 'Unknown device'}), 404

#     accel_mag = math.sqrt(data.get('ax',0)**2 + data.get('ay',0)**2 + data.get('az',0)**2)

#     log = SensorLog(
#         elder_id=elder.id,
#         ax=data.get('ax'), ay=data.get('ay'), az=data.get('az'),
#         gx=data.get('gx'), gy=data.get('gy'), gz=data.get('gz'),
#         accel_mag=accel_mag,
#         lat=data.get('lat'), lng=data.get('lng'),
#         gps_fixed=data.get('gps_fixed', False),
#         battery=data.get('battery', 100)
#     )
#     db.session.add(log)

#     fall_detected = data.get('fall_event', False)
#     confidence    = data.get('confidence', 0)

#     if fall_detected:
#         fall = FallEvent(
#             elder_id=elder.id,
#             confidence=confidence,
#             lat=data.get('lat'), lng=data.get('lng')
#         )
#         db.session.add(fall)
#         db.session.flush()

#         # Notify caretaker
#         if elder.caretaker_id:
#             notif = Notification(
#                 user_id=elder.caretaker_id,
#                 message=f"🚨 Fall detected for {elder.name}! Confidence: {confidence:.1f}%",
#                 type='danger'
#             )
#             db.session.add(notif)

#         # Notify elder
#         elder_notif = Notification(
#             user_id=elder.id,
#             message=f"⚠️ A fall event was detected and your caretaker has been alerted.",
#             type='warning'
#         )
#         db.session.add(elder_notif)
#         db.session.commit()

#         socketio.emit('fall_alert', {
#             'elder_id':   elder.id,
#             'elder_name': elder.name,
#             'id':         fall.id,
#             'confidence': confidence,
#             'lat': data.get('lat'), 'lng': data.get('lng'),
#             'timestamp':  fall.timestamp.isoformat()
#         })
#     else:
#         db.session.commit()

#     socketio.emit(f'sensor_{elder.id}', {
#         'ax': data.get('ax'), 'ay': data.get('ay'), 'az': data.get('az'),
#         'gx': data.get('gx'), 'gy': data.get('gy'), 'gz': data.get('gz'),
#         'accel_mag': accel_mag,
#         'lat': data.get('lat'), 'lng': data.get('lng'),
#         'gps_fixed': data.get('gps_fixed', False),
#         'battery': data.get('battery', 100),
#         'fall_detected': fall_detected,
#         'confidence': confidence,
#         'timestamp': datetime.utcnow().isoformat()
#     })

#     return jsonify({'status': 'ok', 'fall_detected': fall_detected}), 200

# ─────────────────────────────────────────────────────────────
# RANDOM DATA SIMULATOR (dev only — remove in production)
# ─────────────────────────────────────────────────────────────

def simulate_sensor_data():
    """Pushes fake sensor readings every second for all elders."""
    with app.app_context():
        time.sleep(3)  # Wait for DB to be ready
        while True:
            elders = User.query.filter_by(role='elder').all()
            for elder in elders:
                ax = round(random.uniform(-1.2, 1.2), 3)
                ay = round(random.uniform(-1.2, 1.2), 3)
                az = round(random.uniform(0.8, 1.2), 3)
                gx = round(random.uniform(-15, 15), 3)
                gy = round(random.uniform(-15, 15), 3)
                gz = round(random.uniform(-15, 15), 3)
                accel_mag = round(math.sqrt(ax**2 + ay**2 + az**2), 3)

                # Simulate occasional fall (1% chance)
                fall_detected = random.random() < 0.01
                confidence    = round(random.uniform(75, 98), 1) if fall_detected else 0

                log = SensorLog(
                    elder_id=elder.id,
                    ax=ax, ay=ay, az=az,
                    gx=gx, gy=gy, gz=gz,
                    accel_mag=accel_mag,
                    lat=19.0760 + random.uniform(-0.001, 0.001),
                    lng=72.8777 + random.uniform(-0.001, 0.001),
                    gps_fixed=True, battery=random.randint(60, 100)
                )
                db.session.add(log)

                if fall_detected:
                    fall = FallEvent(
                        elder_id=elder.id,
                        confidence=confidence,
                        lat=19.0760, lng=72.8777
                    )
                    db.session.add(fall)
                    db.session.flush()

                    if elder.caretaker_id:
                        db.session.add(Notification(
                            user_id=elder.caretaker_id,
                            message=f"🚨 Fall detected for {elder.name}! Confidence: {confidence}%",
                            type='danger'
                        ))
                    db.session.add(Notification(
                        user_id=elder.id,
                        message="⚠️ Fall detected. Your caretaker has been notified.",
                        type='warning'
                    ))
                    db.session.commit()

                    socketio.emit('fall_alert', {
                        'elder_id': elder.id, 'elder_name': elder.name,
                        'id': fall.id, 'confidence': confidence,
                        'lat': 19.0760, 'lng': 72.8777,
                        'timestamp': fall.timestamp.isoformat()
                    })
                else:
                    db.session.commit()

                socketio.emit(f'sensor_{elder.id}', {
                    'ax': ax, 'ay': ay, 'az': az,
                    'gx': gx, 'gy': gy, 'gz': gz,
                    'accel_mag': accel_mag,
                    'lat': 19.0760, 'lng': 72.8777,
                    'gps_fixed': True, 'battery': log.battery,
                    'fall_detected': fall_detected,
                    'confidence': confidence,
                    'timestamp': datetime.utcnow().isoformat()
                })
            time.sleep(1)

# ─────────────────────────────────────────────────────────────
# SOCKET EVENTS
# ─────────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    if current_user.is_authenticated:
        emit('connected', {'role': current_user.role, 'name': current_user.name})

@socketio.on('subscribe_elder')
def on_subscribe(data):
    pass  # Client sends elder_id; server already emits to sensor_{id}

# ─────────────────────────────────────────────────────────────
# DB INIT + SEED
# ─────────────────────────────────────────────────────────────

def seed_db():
    if User.query.filter_by(email='admin@gs.com').first():
        return  # Already seeded
    admin = User(
        name='Admin',
        email='admin@gs.com',
        password=bcrypt.generate_password_hash('admin123').decode(),
        role='admin'
    )
    care = User(
        name='Nurse Rita',
        email='rita@gs.com',
        password=bcrypt.generate_password_hash('care123').decode(),
        role='caretaker'
    )
    db.session.add_all([admin, care])
    db.session.flush()

    elder = User(
        name='Mr. Sharma',
        email='sharma@gs.com',
        password=bcrypt.generate_password_hash('elder123').decode(),
        role='elder',
        device_id='ESP32-001',
        caretaker_id=care.id
    )
    db.session.add(elder)
    db.session.commit()
    print("Demo users seeded — admin@gs.com / rita@gs.com / sharma@gs.com")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_db()

    # Start simulator in background thread
    #sim_thread = threading.Thread(target=simulate_sensor_data, daemon=True)
    #sim_thread.start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)