import '../styles/Header.css';

const Header = () => {
    return (
    <header className="header-container text-center p-3 bg-light">
        <h1 className="header-title">Music Genre Classifier</h1>
        <p className="header-subtitle">Upload an audio file to see its match percentages to different genres</p>
      </header>
    );
  };

export default Header;