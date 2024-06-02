

'''

FER Falcons.AI Hackathon 224
Team Name: Syntax Squad
Team Leader: Zafar Shaikh
Team Member: Amaal Mecci, Saif Faisal
Email ID: skhzafar110@gmail.com

'''



from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import os
import new_detection as detect
import threading
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
socketio = SocketIO(app)

processing_status = {}


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                result_filename = process_image(file_path)
                return redirect(url_for('show_result', result_image=result_filename))
            elif filename.lower().endswith(('mp4', 'avi', 'mov')):
                result_filename = process_video(file_path)
                return redirect(url_for('show_result', result_video=result_filename))

    return render_template('index.html')


def process_image(image_path):
    result_filename = os.path.basename(image_path)
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    detect.analyze_image(image_path, result_path)
    return result_filename


def process_video(video_path):
    result_filename = os.path.basename(video_path)
    detect.analyze_video(video_path)


@app.route('/result')
def show_result():
    result_image = request.args.get('result_image')
    result_video = request.args.get('result_video')
    return render_template('result.html', result_image=result_image, result_video=result_video)


@app.route('/static/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/static/results/<filename>')
def send_result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == "__main__":
    socketio.run(app, debug=True)
