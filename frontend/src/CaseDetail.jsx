import { useState, useEffect } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import './CaseDetail.css'

const API_URL = 'http://localhost:8000'

function CaseDetail() {
  const { caseIndex } = useParams()
  const navigate = useNavigate()
  const [caseData, setCaseData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchCaseDetail()
  }, [caseIndex])

  const fetchCaseDetail = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.get(`${API_URL}/case/${caseIndex}`)
      setCaseData(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load case details')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="app">
        <header className="header">
          <div className="container">
            <Link to="/" className="logo" style={{ textDecoration: 'none', color: 'inherit' }}>
              <h1>⚖️ AI legal Assistant</h1>
              <p>Nepali Legal Search Engine</p>
            </Link>
          </div>
        </header>
        <main className="main">
          <div className="container">
            <div className="loading-container">
              <div className="spinner-large"></div>
              <p>Loading case details...</p>
            </div>
          </div>
        </main>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app">
        <header className="header">
          <div className="container">
            <Link to="/" className="logo" style={{ textDecoration: 'none', color: 'inherit' }}>
              <h1>⚖️ AI legal assistant</h1>
              <p>Nepali Legal Search Engine</p>
            </Link>
          </div>
        </header>
        <main className="main">
          <div className="container">
            <div className="error-container">
              <h2>❌ Error</h2>
              <p>{error}</p>
              <Link to="/" className="back-button">
                ← Back to Search
              </Link>
            </div>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="container">
          <Link to="/" className="logo" style={{ textDecoration: 'none', color: 'inherit' }}>
            <h1>⚖️ AI legal Assistant</h1>
            <p>Nepali Legal Search Engine</p>
          </Link>
          <button onClick={() => navigate(-1)} className="back-btn-header">
            ← Back
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="main">
        <div className="container-detail">
          <div className="detail-card">
            {/* Back Button and Download */}
            <div className="detail-header">
              <button onClick={() => navigate(-1)} className="back-button">
                ← Back to Results
              </button>
              {caseData.filename && (
                <a 
                  href={`${API_URL}/download/${encodeURIComponent(caseData.filename)}`}
                  download={caseData.filename}
                  className="download-button"
                >
                  📥 Download Original File
                </a>
              )}
            </div>

            {/* Case Title */}
            <div className="case-title-section">
              <h1 className="case-title">Legal Case Details</h1>
              {caseData.case_number_english && (
                <div className="case-number-badge">
                  {caseData.case_number_english}
                </div>
              )}
            </div>

            {/* Case Metadata */}
            <div className="case-metadata-grid">
              {caseData.case_number_nepali && (
                <div className="metadata-item">
                  <div className="metadata-label">📋 Case Number (Nepali)</div>
                  <div className="metadata-value">{caseData.case_number_nepali}</div>
                </div>
              )}

              {caseData.case_type_english && (
                <div className="metadata-item">
                  <div className="metadata-label">⚖️ Case Type</div>
                  <div className="metadata-value">
                    {caseData.case_type_english}
                    {caseData.case_type_nepali && (
                      <span className="metadata-nepali"> ({caseData.case_type_nepali})</span>
                    )}
                  </div>
                </div>
              )}

              {caseData.court_english && (
                <div className="metadata-item">
                  <div className="metadata-label">🏛️ Court</div>
                  <div className="metadata-value">
                    {caseData.court_english}
                    {caseData.court_nepali && (
                      <span className="metadata-nepali"> ({caseData.court_nepali})</span>
                    )}
                  </div>
                </div>
              )}

              {caseData.filename && (
                <div className="metadata-item">
                  <div className="metadata-label">📁 Source File</div>
                  <div className="metadata-value filename">{caseData.filename}</div>
                </div>
              )}
            </div>

            {/* Full Summary with Markdown */}
            {caseData.summary && caseData.summary !== "Summary generation failed" ? (
              <div className="case-full-summary">
                <h2 className="summary-title">📝 Complete Case Summary</h2>
                <div className="markdown-content">
                  <ReactMarkdown
                    components={{
                      h1: ({node, ...props}) => <h3 className="md-h1" {...props} />,
                      h2: ({node, ...props}) => <h4 className="md-h2" {...props} />,
                      h3: ({node, ...props}) => <h5 className="md-h3" {...props} />,
                      p: ({node, ...props}) => <p className="md-p" {...props} />,
                      ul: ({node, ...props}) => <ul className="md-ul" {...props} />,
                      ol: ({node, ...props}) => <ol className="md-ol" {...props} />,
                      li: ({node, ...props}) => <li className="md-li" {...props} />,
                      strong: ({node, ...props}) => <strong className="md-strong" {...props} />,
                      em: ({node, ...props}) => <em className="md-em" {...props} />,
                      blockquote: ({node, ...props}) => <blockquote className="md-blockquote" {...props} />,
                      code: ({node, inline, ...props}) => 
                        inline ? <code className="md-code-inline" {...props} /> : <code className="md-code-block" {...props} />
                    }}
                  >
                    {caseData.summary}
                  </ReactMarkdown>
                </div>
              </div>
            ) : (
              <div className="no-summary-available">
                <p>ℹ️ Detailed summary not available for this case</p>
              </div>
            )}
          </div>
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

export default CaseDetail

