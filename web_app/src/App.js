import { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import Results from './components/Results';

function App() {
  const [results, setResults] = useState([]);

  return (
    <div>
      <Header />
      <FileUpload setResults={setResults}/>
      <Results results={results} />
    </div>
  );
}

export default App;
