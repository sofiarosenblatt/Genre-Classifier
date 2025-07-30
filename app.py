import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import yt_dlp
import torch
from dotenv import load_dotenv
import os

from utils.utils import load_file, extract_spectrogram
from utils.CRNN import CRNN

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://genre-classifier-app.onrender.com/", "https://genre-classifier-app.onrender.com"])

load_dotenv()
temp_dir = os.environ.get('TEMP_DIR', './temp')
os.makedirs(temp_dir, exist_ok=True)

model = CRNN(input_channels=1, num_classes=10, rnn_hidden_size=128, rnn_layers=1, dropout=0.1)
model.load_state_dict(torch.load("model/crnn_state_dict.pth", weights_only=True, map_location=torch.device('cpu')))
model.eval()

def download_audio(url):
    """Downloads audio from YouTube and returns the file path."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'cookiefile': 'YTCookies.txt',
        'writethumbnail': False,
        'noplaylist': True,
        'writedescription': False,
        'outtmpl': f'{temp_dir}/%(title)s.%(ext)s',
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': 192,
            },
            {
                'actions': [
                    (yt_dlp.postprocessor.metadataparser.MetadataParserPP.replacer, 'title', r"[^\v\n\S]", "_"),
                    (yt_dlp.postprocessor.metadataparser.MetadataParserPP.replacer, 'title', r"[^A-Za-z0-9_]", "")
                ],
                'key': 'MetadataParser',
                'when': 'pre_process'                
            }
        ],
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = f"{temp_dir}/{info_dict['title']}.wav"
        return filename

@app.route('/api/predict', methods=['POST'])
def predict():

    # Check for YouTube URL or uploaded file in POST request
    youtube_url = request.form.get('youtube_url')
    file = request.files.get('file')

    if not youtube_url and (not file or not file.filename):
        return jsonify({'error': 'No file or YouTube URL provided'}), 400
    
    if youtube_url:
        # Download audio from YouTube
        try:
            youtube_url = youtube_url.split("&list")[0]
            file_path = download_audio(youtube_url)
        except Exception as e:
            return jsonify({'error': f'YouTube download failed: {str(e)}'}), 500
    else:    
        # Verify valid audio file format
        if file.filename.split('.')[-1].lower() not in ('wav', 'mp3', 'aiff'):
            return jsonify({'error': f'{file.filename} is not a valid audio file'}), 400
    
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

    try:
        # Load audio
        signal, sr = load_file(file_path)
        # Extract spectrogram and add batch dimension
        spectrogram = extract_spectrogram(signal, sr).unsqueeze(0)

        # Predict
        with torch.no_grad():
            prediction = model(spectrogram)
            prediction = torch.nn.functional.softmax(prediction, -1)[0].tolist()
        return jsonify(prediction)
    finally:
        # Remove temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
