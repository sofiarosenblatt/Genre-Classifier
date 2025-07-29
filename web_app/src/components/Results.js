import '../styles/Results.css';

const Results = ({ results }) => {
  const colors = [
    '#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1',
    '#955251', '#B565A7', '#009B77', '#DD4124', '#45B8AC',
  ]; 

  const genres = {
    0: "Blues", 
    1: "Classical", 
    2: "Country", 
    3: "Disco", 
    4: "Hiphop", 
    5: "Jazz", 
    6: "Metal", 
    7: "Pop", 
    8: "Reggae",
    9: "Rock"
  }

  if (!results || results.length === 0) {
    return null; // Do not render if no predictions are provided
  }

  return (
    <section className="results-container py-5 text-center bg-light">
      <div className="container">
        <h3>Results</h3>
        <ul className="results-bars">
          {results.map((value, index) => (
            <li key={index} className="results-bar">
              <span className="label">{`${genres[index]}`}</span>
              <div className="bar-container">
                <div
                  className="bar"
                  style={{
                    width: `${value * 100}%`,
                    backgroundColor: colors[index % colors.length],
                  }}
                ></div>
                <span className="percentage">{Math.round(value * 100)}%</span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
};

export default Results;
