import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import './App.css'

const API_URL = 'http://localhost:8000'

function SearchPage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)
  const [searched, setSearched] = useState(false)

  // Fetch statistics on load
  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/stats`)
      setStats(response.data)
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    }
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    
    if (!query.trim()) {
      setError('Please enter a search query')
      return
    }

    setLoading(true)
    setError(null)
    setSearched(true)

    try {
      const response = await axios.post(`${API_URL}/search`, {
        query: query.trim(),
        limit: 5
      })

      setResults(response.data.results)
      
      if (response.data.results.length === 0) {
        setError('No results found. Try a different query.')
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Search failed. Please try again.')
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const exampleQueries = [
    'murder case',
    'property dispute',
    'inheritance rights',
    'constitutional matter',
    'corruption case'
  ]

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="container">
          <Link to="/" className="logo" style={{ textDecoration: 'none', color: 'inherit' }}>
            <h1>AI legal Assistant</h1>
            <p>Nepali Legal Search Engine</p>
          </Link>
          <nav className="nav-tabs">
            <Link to="/" className="nav-tab active">🔍 Search</Link>
            <Link to="/strategy" className="nav-tab">⚔️ CaseFighting</Link>
            <Link to="/chat" className="nav-tab">🤖 AI Assistant</Link>
          </nav>
          {stats && (
            <div className="stats-badge">
              <span>{stats.total_cases.toLocaleString()} Cases</span>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="main">
        <div className="container">
          {/* Hero Section */}
          <div className="hero">
            <h2>AI-Powered Semantic Search for Nepali Legal Cases</h2>
            <p>Search through 10,000+ court cases in English or Nepali using natural language</p>
          </div>

          {/* Search Box */}
          <form onSubmit={handleSearch} className="search-form">
            <div className="search-box">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search for legal cases... (e.g., 'murder case' or 'सम्पत्ति विवाद')"
                className="search-input"
                disabled={loading}
              />
              <button 
                type="submit" 
                className="search-button"
                disabled={loading}
              >
                {loading ? (
                  <span className="spinner"></span>
                ) : (
                  '🔍 Search'
                )}
              </button>
            </div>
          </form>

          {/* Example Queries */}
          {!searched && (
            <div className="examples">
              <p>Try searching for:</p>
              <div className="example-tags">
                {exampleQueries.map((example, index) => (
                  <button
                    key={index}
                    className="example-tag"
                    onClick={() => setQuery(example)}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

          {/* Results */}
          {searched && !loading && results.length > 0 && (
            <div className="results">
              <div className="results-header">
                <h3>Search Results</h3>
                <p>{results.length} cases found</p>
              </div>

              {results.map((result) => (
                <div key={result.rank} className="result-card">
                  <div className="result-header">
                    <div className="result-rank">#{result.rank}</div>
                    <div className="result-score">
                      Relevance: {(result.score * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="result-body">
                    {/* Case Info */}
                    <div className="case-info">
                      {result.case_number_english && (
                        <div className="info-item">
                          <span className="info-label">📋 Case Number:</span>
                          <span className="info-value">{result.case_number_english}</span>
                          {result.case_number_nepali && (
                            <span className="info-value-nepali">({result.case_number_nepali})</span>
                          )}
                        </div>
                      )}

                      {result.case_type_english && (
                        <div className="info-item">
                          <span className="info-label">⚖️ Case Type:</span>
                          <span className="info-value">{result.case_type_english}</span>
                          {result.case_type_nepali && (
                            <span className="info-value-nepali">({result.case_type_nepali})</span>
                          )}
                        </div>
                      )}

                      {result.court_english && (
                        <div className="info-item">
                          <span className="info-label">🏛️ Court:</span>
                          <span className="info-value">{result.court_english}</span>
                          {result.court_nepali && (
                            <span className="info-value-nepali">({result.court_nepali})</span>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Summary Preview */}
                    {result.summary && result.summary !== "Summary generation failed" && (
                      <div className="case-summary">
                        <h4>📝 Summary Preview</h4>
                        <p>{result.summary.substring(0, 300)}{result.summary.length > 300 ? '...' : ''}</p>
                      </div>
                    )}

                    {/* View Details Button */}
                    <div className="case-actions">
                      <Link to={`/case/${result.index}`} className="view-details-btn">
                        📄 View Full Details
                      </Link>
                    </div>

                    {/* Filename */}
                    <div className="case-file">
                      <span className="file-label">📁 Source:</span>
                      <span className="file-name">{result.filename}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* No Results */}
          {searched && !loading && results.length === 0 && !error && (
            <div className="no-results">
              <p>No results found for "{query}"</p>
              <p>Try using different keywords or rephrasing your query</p>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>© 2026 Powered by AI & Semantic Search</p>
          <p>Built with ❤️ for accessible legal information in Nepal</p>
        </div>
      </footer>
    </div>
  )
}

export default SearchPage

