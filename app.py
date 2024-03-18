from flask import Flask, render_template, Response, request,redirect,url_for
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from face_recognition import face_encodings, load_image_file,compare_faces,face_locations,face_distance
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import time

app = Flask(__name__, static_url_path='/static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///person.db'
db = SQLAlchemy(app)
socketio = SocketIO(app)

class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    address = db.Column(db.String(100))
    mobile = db.Column(db.String(10))
    email = db.Column(db.String(100))
    image = db.Column(db.String(100))
    face_encoding = db.Column(db.PickleType)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        address = request.form['address']
        mobile = request.form['mobile']
        email = request.form['email']
        image = request.files['image']
        image_path = "static/images/" + image.filename
        image.save(image_path)  # Save image to a folder
        face_image = load_image_file(image_path)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_location = face_locations(face_image)
        face_enc = face_encodings(face_image, face_location)[0]
        new_person = Person(name=name, address=address, mobile=mobile, email=email, image=image.filename, face_encoding=face_enc)
        db.session.add(new_person)
        db.session.commit()
        print("New person added to database:", new_person)  # Debugging statement
        # return redirect(url_for('index'))
        return redirect(url_for('index1'))

@app.route('/stored_data')
def stored_data():
    persons = Person.query.all()
    return render_template('stored_data.html', persons=persons)




    # Function to process incoming frames for face recognition
def process_frame(data):
    with app.app_context():
        known_encodings = []
        persons = Person.query.all()
        for person in persons:
            known_encodings.append(np.array(person.face_encoding))
        tolerance = 0.6
        base64_data = data['data']

        image_data = base64.b64decode(base64_data)
            
        image = np.array(Image.open(BytesIO(image_data)))
            
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_location = face_locations(small_frame)
        face_encoding = face_encodings(small_frame, face_location)

            # Iterate over each face found in the frame
        for face in face_encoding:
                # Compare the current face encoding with known encodings from the database
            matches = compare_faces(known_encodings, face)
            face_distances = face_distance(known_encodings, face)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                matched_person = persons[best_match_index]
                print("Match found:", matched_person.name) 
                socketio.emit('match_found', {
                    'name': matched_person.name, 
                    'address': matched_person.address,
                    'mobile': matched_person.mobile, 
                    'email': matched_person.email, 
                    'image': matched_person.image})
                
            else:
                socketio.emit('no_match')
                
          
# WebSocket endpoint to receive video frames
@socketio.on('frame')
def handle_frame(data):
    process_frame(data)



# @socketio.on('stop')
# def stop_stream():
#     global video_stream
#     if video_stream is not None:
#         video_stream.release()
#         video_stream = None


if __name__ == '__main__':
    with app.app_context():  # Ensure operations are within application context
        db.create_all()
        print("Database tables created successfully")
    socketio.run(app)


