from flask import Flask
from flask_cors import CORS
from modules.face_recognition import face_recognition_bp


app = Flask(__name__)
CORS(app)

app.register_blueprint(face_recognition_bp)

if __name__ == '__main__':
    app.run(debug=True)
