from flask import Flask, render_template, request, redirect, url_for, flash
import requests
from PIL import Image
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Для отображения сообщений flash
app.config['UPLOAD_FOLDER'] = 'flask_app/static/uploads'

# Создаем папку для загрузок, если она не существует
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    response = requests.get('http://fastapi:8000')
    root_data = response.json()
    root_message = root_data.get('Hello', 'No message')

    pytorch_response = requests.get('http://fastapi:8000/pytorch')
    pytorch_data = pytorch_response.json()
    pytorch_message = str(pytorch_data)

    return render_template('index.html', root_message=root_message, pytorch_message=pytorch_message)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        try:
            img = Image.open(file)
            if img.size[0] > 512 or img.size[1] > 512:
                flash('Image dimensions should not exceed 512x512 pixels')
                return redirect(request.url)

            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(file_path)

            # Отправка файла в FastAPI для предсказания
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f, 'image/jpeg')}
                response = requests.post('http://fastapi:8000/predict', files=files)
                if response.status_code == 200:
                    prediction = response.json().get('predicted_class')
                    flash('File successfully uploaded')
                    return render_template('index.html', prediction=prediction, filename=filename)
                else:
                    flash('Error in prediction')
                    return redirect(request.url)
        except Exception as e:
            flash('Error processing image')
            return redirect(request.url)
    else:
        flash('Allowed file types are jpg, jpeg, png')
        return redirect(request.url)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)