import { useState } from 'react';
import axios from 'axios';
import '../styles/FileUpload.css'; 

const youtube_api_key = process.env.REACT_APP_YOUTUBE_API_KEY;
const api_base_url = process.env.REACT_APP_API_BASE_URL;

const FileUpload = ({ setResults }) => {
  const [youtubeLink, setYoutubeLink] = useState('');
  const [videoId, setVideoId] = useState(null);
  const [videoTitle, setVideoTitle] = useState('');
  const [fileName, setFileName] = useState('');
  const [loading, setLoading] = useState(false);
  const [ytErrorMessage, setYtErrorMessage] = useState('');
  const [fileErrorMessage, setFileErrorMessage] = useState('');

  const extractYouTubeID = (url) => {
    if ( url.match("youtube.com/") ) {
      url = url.split("outube.com/")[1];

      if ( url.match(/[?&]v=/) ) {
        return url.split("v=")[1].substring(0, 11);
      } 
      else if ( url.match(/v|e(?:mbed)\//) || url.match(/shorts\//) ) {
        return url.split("/")[1].substring(0, 11);
      }
    } 
    else if ( url.match("youtu.be") ) {
      url = url.split("outu.be/")[1];
      return url.substring(0, 11);
    } 
    else {
      return null;
    }
  };

  const validateYouTubeVideo = async (videoID) => {
    try {
      const response = await axios.get(
        `https://www.googleapis.com/youtube/v3/videos?part=id&id=${videoID}&key=${youtube_api_key}&part=status,snippet`
      );
      if (response.data.items.length === 0) {
        setYtErrorMessage('This video is unavailable.');
        setVideoId(null);
        setVideoTitle('');
      } else {
        setYtErrorMessage('');
        setVideoTitle(response.data.items[0].snippet.title); // Store video title
        setVideoId(videoID);
      }
    } catch (error) {
      console.error('YouTube API Error:', error);
      setYtErrorMessage('Failed to validate the video.');
      setVideoId(null);
      setVideoTitle('');
    }
  };

  const handleYoutubeChange = (event) => {
    var url = event.target.value;
    if (url && !url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'https://' + url;
    }
    setYoutubeLink(url);
    setFileName(''); // Clear file selection if a YouTube link is pasted
    setFileErrorMessage('');
    setResults('');
    document.getElementById("fileInput").value = '';

    const videoID = extractYouTubeID(url);
    if (videoID) {
      validateYouTubeVideo(videoID);
    } else {
      setVideoId(null);
      setYtErrorMessage('Invalid YouTube link. Please enter a valid URL.');
    }
  };
  
  const handleFileChange = (event) => {
    if (event.target.files.length > 0) {
      setFileName(event.target.files[0].name); // Update the state with the file name
      setYoutubeLink(''); // Clear YouTube link and ID if a file is chosen
      setVideoId(null);
      setYtErrorMessage('');
      setVideoTitle('');
      setResults('');
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (fileName.trim().length === 0 && youtubeLink.trim().length === 0) {
      alert('Please select a file or enter a YouTube link.');
      return;
    }

    const formData = new FormData();
    if (youtubeLink.trim().length !== 0) {
      if (!videoId) {
        alert('Please enter a valid YouTube link.');
        return;
      }
      formData.append('youtube_url', youtubeLink);
    } else {
      const fileInput = document.getElementById('fileInput');
      formData.append('file', fileInput.files[0]);
    }

    setLoading(true); // Start loading
    try {
      const response = await axios.post(`${api_base_url}/api/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResults(response.data); // Pass predictions to the parent component
    } catch (error) {
      console.error('Error during prediction:', error);
      alert('An error occurred. Please try again.');
    } finally {
        setLoading(false); // Stop loading
      }
  };

  return (
    <section className="file-upload-section py-5 text-center bg-light">
      <div className="container">
        <form onSubmit={handleSubmit}>
          {/* YouTube Link Input */}
          <div className="mb-3">
              <input
                type="url"
                className="form-control form-control-lg"
                placeholder="Enter a YouTube link"
                id="ytLinkInput"
                value={youtubeLink}
                onChange={handleYoutubeChange}
              />
            </div>

          {/* Error Message for Invalid YouTube Links */}
          {ytErrorMessage && <p className="text-danger">{ytErrorMessage}</p>}

          {/* YouTube Video Preview */}
          {videoId && (
            <div className="video-preview">
              <div>
                <img
                  src={`https://img.youtube.com/vi/${videoId}/hqdefault.jpg`}
                  alt="YouTube Video Preview"
                  className="youtube-thumbnail"
                />
              </div>
              <div>
                <p><strong>{videoTitle}</strong></p>
              </div>
            </div>
          )}

          <h3>OR</h3>

          {/* File Upload Input */}
          <div className="mb-3">
            <input
              type="file"
              className="form-control form-control-lg"
              id="fileInput"
              accept="audio/*"
              onChange={handleFileChange}
            />
          </div>

          {/* Error Message for Invalid File */}
          {fileErrorMessage && <p className="text-danger">{fileErrorMessage}</p>}

          {/* Submit Button */}
          <button type="submit" className="btn btn-primary btn-lg" disabled={loading}>
            {loading ? 'Processing...' : 'Submit'}
          </button>
        </form>

        {/* Loading Indicator */}
        {loading && (
          <div className="loading-container mt-4">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
            <p>Generating matches, please wait...</p>
          </div>
        )}
      </div>
    </section>
  );
};

export default FileUpload;
