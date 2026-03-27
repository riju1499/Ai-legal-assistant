import { useState } from 'react'
import { Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import './StrategyPage.css'

function StrategyPage() {
  const [caseFacts, setCaseFacts] = useState('')
  const [desiredOutcome, setDesiredOutcome] = useState('')
  const [caseType, setCaseType] = useState('')
  const [includeRetrieval, setIncludeRetrieval] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [strategy, setStrategy] = useState(null)
  const [agenticInfo, setAgenticInfo] = useState(null)

  const handleGenerate = async () => {
    setError('')
    setStrategy(null)
    if (!caseFacts.trim()) {
      setError('Please enter case facts.')
      return
    }
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/strategy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_facts: caseFacts,
          desired_outcome: desiredOutcome || null,
          case_type: caseType || null,
          include_retrieval: includeRetrieval
        })
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Request failed: ${res.status}`)
      }
      const data = await res.json()
      setStrategy(data.strategy || data)
      // Extract agentic reasoning info if available
      if (data.strategy?.agentic_reasoning) {
        setAgenticInfo(data.strategy.agentic_reasoning)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const renderSummary = (s) => {
    if (!s || typeof s !== 'object') return null
    const sp = s.success_probability || {}
    const para = s.strategic_paragraph
    return (
      <div className="strategy-markdown">
        <h3>Strategy Summary</h3>
        <div><b>Case Type:</b> {s.case_type || 'Unknown'}</div>
        {s.desired_outcome ? (<div><b>Desired Outcome:</b> {s.desired_outcome}</div>) : null}
        {para ? (
          <div style={{ marginTop: 8 }}>
            <b>Strategic Plan</b>
            <div>{para}</div>
          </div>
        ) : null}
        {sp.point !== undefined ? (
          <div className="success-probability">
            🎯 Success Probability: {Number(sp.point * 100).toFixed(0)}% 
            {Array.isArray(sp.ci) && sp.ci.length === 2 ? ` (${Number(sp.ci[0] * 100).toFixed(0)}% - ${Number(sp.ci[1] * 100).toFixed(0)}%)` : ''}
          </div>
        ) : null}
        {Array.isArray(s.applicable_laws) && s.applicable_laws.length ? (
          <div className="strategy-section">
            <h4>⚖️ Applicable Laws</h4>
            <ul>
              {s.applicable_laws.slice(0,5).map((l, i) => (
                <li key={i}>
                  <strong>{l.section}</strong>
                  <br />
                  <span style={{ color: '#666', fontSize: '0.95rem' }}>{l.why}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {Array.isArray(s.arguments) && s.arguments.length ? (
          <div className="strategy-section">
            <h4>💬 Key Arguments</h4>
            <ul>
              {s.arguments.slice(0,5).map((a, i) => (
                <li key={i}>
                  <strong>{a.claim}</strong>
                  {Array.isArray(a.support) && a.support.length > 0 && (
                    <ul style={{ marginTop: '0.5rem', paddingLeft: '1rem' }}>
                      {a.support.slice(0,3).map((sup, j) => (
                        <li key={j} style={{ fontSize: '0.9rem', color: '#666', borderLeft: 'none', background: 'none', padding: '0.25rem 0' }}>
                          • {sup}
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {Array.isArray(s.documents_checklist) && s.documents_checklist.length ? (
          <div className="strategy-section">
            <h4>📄 Documents to Prepare</h4>
            <ul>
              {s.documents_checklist.slice(0,8).map((d, i) => {
                const priority = d.priority?.toLowerCase() || '';
                const priorityClass = priority === 'high' ? 'priority-high' : 
                                     priority === 'medium' ? 'priority-medium' : 
                                     priority === 'low' ? 'priority-low' : '';
                return (
                  <li key={i}>
                    <strong>{d.document}</strong>
                    <br />
                    <span style={{ color: '#666', fontSize: '0.95rem' }}>
                      {d.purpose}
                      {d.required_from && ` (Source: ${d.required_from})`}
                    </span>
                    {d.priority && <span className={priorityClass}>{d.priority.toUpperCase()}</span>}
                  </li>
                );
              })}
            </ul>
          </div>
        ) : null}
        {Array.isArray(s.counter_arguments) && s.counter_arguments.length ? (
          <div className="strategy-section">
            <h4>🛡️ Counter-Arguments & Responses</h4>
            <ul>
              {s.counter_arguments.slice(0,3).map((c, i) => (
                <li key={i}>
                  <strong style={{ color: '#e53935' }}>Opponent: {c.claim}</strong>
                  <br />
                  <span style={{ color: '#333', marginTop: '0.5rem', display: 'block' }}>
                    <strong>Response:</strong> {c.response}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {Array.isArray(s.winning_points) && s.winning_points.length ? (
          <div className="strategy-section">
            <h4>🏆 Points to Emphasize to Win</h4>
            <ul>
              {s.winning_points.slice(0,6).map((wp, i) => (
                <li key={i} style={{ background: 'linear-gradient(90deg, #fff5e6 0%, #fff 100%)', borderLeftColor: '#ffa726' }}>
                  {wp}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {(Array.isArray(s.strengths) && s.strengths.length) || (Array.isArray(s.weaknesses) && s.weaknesses.length) ? (
          <div className="strategy-section">
            {Array.isArray(s.strengths) && s.strengths.length > 0 && (
              <div style={{ marginBottom: '1rem' }}>
                <h4 style={{ color: '#66bb6a', borderLeftColor: '#66bb6a' }}>✅ Strengths</h4>
                <ul>
                  {s.strengths.slice(0,5).map((st, i) => (
                    <li key={i} style={{ background: '#f1f8f4', borderLeftColor: '#66bb6a' }}>
                      {st}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {Array.isArray(s.weaknesses) && s.weaknesses.length > 0 && (
              <div>
                <h4 style={{ color: '#e53935', borderLeftColor: '#e53935' }}>⚠️ Weaknesses</h4>
                <ul>
                  {s.weaknesses.slice(0,5).map((wk, i) => (
                    <li key={i} style={{ background: '#fff5f5', borderLeftColor: '#e53935' }}>
                      {wk}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : null}
        {Array.isArray(s.procedural_timeline) && s.procedural_timeline.length > 0 && (
          <div className="strategy-section">
            <h4>📅 Procedural Timeline</h4>
            <ul>
              {s.procedural_timeline.slice(0,6).map((t, i) => (
                <li key={i} style={{ background: 'linear-gradient(90deg, #e3f2fd 0%, #fff 100%)', borderLeftColor: '#2196f3' }}>
                  <strong>{t.step || 'Step ' + (i + 1)}</strong>
                  {t.deadline && (
                    <span style={{ color: '#2196f3', marginLeft: '0.5rem', fontWeight: 600 }}>
                      {' → '}{t.deadline}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
        {Array.isArray(s.witness_plan) && s.witness_plan.length > 0 && (
          <div className="strategy-section">
            <h4>👥 Witness Plan</h4>
            <ul>
              {s.witness_plan.slice(0,5).map((w, i) => (
                <li key={i}>
                  <strong>{w.type || 'Witness ' + (i + 1)}</strong>
                  {w.goal && (
                    <>
                      <br />
                      <span style={{ color: '#666', fontSize: '0.95rem' }}>{w.goal}</span>
                    </>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="strategy-container">
      <header className="strategy-header">
        <div className="strategy-inner">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <h2 className="section-title">Case Fighting Agent</h2>
            <Link to="/" className="nav-tab" style={{ textDecoration: 'none' }}>← Home</Link>
          </div>
          <p className="muted">Prepare a causal, precedent-backed strategy with arguments, counter-arguments, evidence plan, and what-ifs.</p>
        </div>
      </header>
      <main className="strategy-main">
        <div className="strategy-inner grid">
          <div className="card">
            <div className="grid">
              <div>
                <div className="label">Case Facts</div>
                <textarea
                  rows={10}
                  className="textarea"
                  value={caseFacts}
                  onChange={(e) => setCaseFacts(e.target.value)}
                  placeholder="Describe the facts, timeline, key evidence, and goals"
                />
              </div>

              <div className="grid-2">
                <div>
                  <div className="label">Desired Outcome (optional)</div>
                  <input
                    type="text"
                    className="input"
                    value={desiredOutcome}
                    onChange={(e) => setDesiredOutcome(e.target.value)}
                    placeholder="e.g., Equal share, Settlement, Injunction"
                  />
                </div>
                <div>
                  <div className="label">Case Type (optional)</div>
                  <input
                    type="text"
                    className="input"
                    value={caseType}
                    onChange={(e) => setCaseType(e.target.value)}
                    placeholder="e.g., Property Dispute, Inheritance, Fraud"
                  />
                </div>
              </div>

              <label className="muted" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <input type="checkbox" checked={includeRetrieval} onChange={(e) => setIncludeRetrieval(e.target.checked)} />
                Include top retrieved cases as context
              </label>

              <div>
                <button className="primary-btn" onClick={handleGenerate} disabled={loading}>
                  {loading ? 'Generating…' : 'Generate Strategy'}
                </button>
              </div>
            </div>
          </div>

          {error ? (
            <div className="card" style={{ borderLeft: '4px solid #c62828' }}>
              <div className="muted" style={{ color: '#c62828' }}>{error}</div>
            </div>
          ) : null}

          {strategy ? (
            <div className="card summary">
              {agenticInfo && (
                <div className="agentic-badge">
                  <strong>🤖 Agentic AI Reasoning:</strong> Used {agenticInfo.iterations} iteration(s) | 
                  Tools: {agenticInfo.tools_used?.join(', ') || 'None'}
                </div>
              )}
              {/* Prefer pretty_markdown if available; otherwise show concise fallback */}
              {strategy.pretty_markdown ? (
                <div className="strategy-markdown">
                  <ReactMarkdown>{strategy.pretty_markdown}</ReactMarkdown>
                </div>
              ) : (
                <div className="strategy-markdown">
                  <h3>Strategy Summary</h3>
                  <div><b>Case Type:</b> {strategy.case_type || 'Unknown'}</div>
                  {strategy.desired_outcome ? (<div><b>Desired Outcome:</b> {strategy.desired_outcome}</div>) : null}
                  {strategy.success_probability && strategy.success_probability.point !== undefined ? (
                    <div className="success-probability">
                      🎯 Success Probability: {Number(strategy.success_probability.point * 100).toFixed(0)}%
                      {Array.isArray(strategy.success_probability.ci) && strategy.success_probability.ci.length === 2 ? ` (${Number(strategy.success_probability.ci[0] * 100).toFixed(0)}% - ${Number(strategy.success_probability.ci[1] * 100).toFixed(0)}%)` : ''}
                    </div>
                  ) : null}
                  {strategy.strategic_paragraph ? (
                    <div style={{ marginTop: 8 }}>
                      <b>Strategic Plan</b>
                      <div>{strategy.strategic_paragraph}</div>
                    </div>
                  ) : null}
                </div>
              )}
              <details style={{ marginTop: 12 }}>
                <summary>View full JSON</summary>
                <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(strategy, null, 2)}</pre>
              </details>
            </div>
          ) : null}
        </div>
      </main>
    </div>
  )
}

export default StrategyPage


