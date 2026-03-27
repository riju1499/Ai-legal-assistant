import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import axios from 'axios'
import './ChatPage.css'

const API_URL = 'http://localhost:8000'
const CHAT_STORAGE_KEY = 'wakalat_sewa_chat_history'
const SESSION_ID_KEY = 'wakalat_sewa_session_id'

function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [chatStatus, setChatStatus] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const messagesEndRef = useRef(null)

  // Load chat history from localStorage on mount
  useEffect(() => {
    loadChatHistory()
    checkChatStatus()
  }, [])

  // Save chat history whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      saveChatHistory()
    }
  }, [messages])

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom()
  }, [messages, loading])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadChatHistory = () => {
    try {
      // Load session ID
      let storedSessionId = localStorage.getItem(SESSION_ID_KEY)
      if (!storedSessionId) {
        storedSessionId = generateSessionId()
        localStorage.setItem(SESSION_ID_KEY, storedSessionId)
      }
      setSessionId(storedSessionId)

      // Load chat messages
      const savedMessages = localStorage.getItem(CHAT_STORAGE_KEY)
      if (savedMessages) {
        const parsed = JSON.parse(savedMessages)
        // Convert timestamp strings back to Date objects
        const messagesWithDates = parsed.map(msg => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }))
        setMessages(messagesWithDates)
      }
    } catch (err) {
      console.error('Failed to load chat history:', err)
    }
  }

  const saveChatHistory = () => {
    try {
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages))
    } catch (err) {
      console.error('Failed to save chat history:', err)
    }
  }

  const generateSessionId = () => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  const clearChatHistory = () => {
    if (window.confirm('Are you sure you want to clear the chat history?')) {
      setMessages([])
      localStorage.removeItem(CHAT_STORAGE_KEY)
      // Generate new session ID
      const newSessionId = generateSessionId()
      setSessionId(newSessionId)
      localStorage.setItem(SESSION_ID_KEY, newSessionId)
    }
  }

  const checkChatStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/chat/status`)
      setChatStatus(response.data)
    } catch (err) {
      console.error('Failed to check chat status:', err)
      setChatStatus({ available: false })
    }
  }

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')

    // Add user message to chat
    const newMessages = [...messages, {
      role: 'user',
      content: userMessage,
      timestamp: new Date()
    }]
    setMessages(newMessages)
    setLoading(true)

    try {
      // Build conversation history (last 10 messages for context)
      const conversationHistory = messages.slice(-10).map(msg => ({
        role: msg.role,
        content: msg.content
      }))

      const response = await axios.post(`${API_URL}/chat`, {
        message: userMessage,
        conversation_id: sessionId,
        conversation_history: conversationHistory
      })

      // Add AI response to chat
      setMessages([...newMessages, {
        role: 'assistant',
        content: response.data.response,
        query_type: response.data.query_type,
        similar_cases: response.data.similar_cases || [],
        causal_explanation: response.data.causal_explanation || '',
        disclaimer: response.data.disclaimer,
        timestamp: new Date()
      }])

    } catch (err) {
      console.error('Chat error:', err)
      setMessages([...newMessages, {
        role: 'error',
        content: err.response?.data?.detail || 'Failed to get response. Please try again.',
        timestamp: new Date()
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const exampleQueries = [
    "What is Article 12 of Nepal Constitution?",
    "I need help with a property dispute case",
    "What happens if someone doesn't register property documents?",
    "Explain inheritance rights for daughters in Nepal",
    "Find cases related to constitutional violations"
  ]

  const handleExampleClick = (query) => {
    setInput(query)
  }

  return (
    <div className="chat-container">
      {/* Header */}
      <header className="chat-header">
        <div className="header-content">
          <Link to="/" className="back-link">
            ← Back to Search
          </Link>
          <div className="header-title">
            <h1>⚖️ AI Legal Assistant</h1>
            <p>Ask me anything about Nepali law</p>
          </div>
          <div className="header-actions">
            {messages.length > 0 && (
              <button onClick={clearChatHistory} className="clear-chat-btn" title="Clear chat history">
                🗑️ Clear Chat
              </button>
            )}
            {chatStatus && (
              <div className={`status-badge ${chatStatus.available ? 'online' : 'offline'}`}>
                {chatStatus.available ? '🟢 Online' : '🔴 Offline'}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="chat-main">
        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">🤖</div>
              <h2>Welcome to Wakalat Sewa AI Assistant</h2>
              <p>I can help you with:</p>
              <ul>
                <li>Explaining Nepali laws and constitutional articles</li>
                <li>Finding similar legal cases and precedents</li>
                <li>Analyzing legal situations with causal reasoning</li>
                <li>Identifying potential legal arguments</li>
              </ul>
              <div className="example-queries">
                <h3>Try asking:</h3>
                {exampleQueries.map((query, idx) => (
                  <button
                    key={idx}
                    className="example-query"
                    onClick={() => handleExampleClick(query)}
                  >
                    {query}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-header">
                    <span className="message-role">
                      {msg.role === 'user' ? '👤 You' : msg.role === 'assistant' ? '🤖 AI Assistant' : '⚠️ Error'}
                    </span>
                    <span className="message-time">
                      {msg.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  
                  <div className="message-content">
                    {msg.role === 'assistant' ? (
                      <>
                        <div className="response-text">
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>

                        {/* Causal Explanation */}
                        {msg.causal_explanation && (
                          <div className="causal-section">
                            <ReactMarkdown>{msg.causal_explanation}</ReactMarkdown>
                          </div>
                        )}

                        {/* Similar Cases */}
                        {msg.similar_cases && msg.similar_cases.length > 0 && (
                          <div className="similar-cases-section">
                            <h4>📚 Similar Cases Referenced:</h4>
                            {msg.similar_cases.map((caseItem, caseIdx) => (
                              <div key={caseIdx} className="case-card-small">
                                <div className="case-header">
                                  <span className="case-number">
                                    {caseItem.case_number_english || 'N/A'}
                                  </span>
                                  <span className="case-score">
                                    Relevance: {(caseItem.score * 100).toFixed(0)}%
                                  </span>
                                </div>
                                <div className="case-info">
                                  <span className="case-type">{caseItem.case_type_english}</span>
                                  <span className="case-court">{caseItem.court_english}</span>
                                </div>
                                <Link to={`/case/${caseItem.index}`} className="view-case-link">
                                  View Full Case →
                                </Link>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Disclaimer */}
                        {msg.disclaimer && (
                          <div className="disclaimer">
                            <ReactMarkdown>{msg.disclaimer}</ReactMarkdown>
                          </div>
                        )}
                      </>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="message assistant loading">
                  <div className="message-header">
                    <span className="message-role">🤖 AI Assistant</span>
                  </div>
                  <div className="message-content">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <p>Analyzing your query...</p>
                  </div>
                </div>
              )}
              
              {/* Scroll anchor */}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </main>

      {/* Input Area */}
      <footer className="chat-input-area">
        {!chatStatus?.available && (
          <div className="warning-banner">
            ⚠️ AI Assistant is currently unavailable. Please ensure the backend is running with Gemini API or Ollama configured.
          </div>
        )}
        <div className="input-container">
          <textarea
            className="chat-input"
            placeholder="Ask a legal question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading || !chatStatus?.available}
            rows={2}
          />
          <button
            className="send-button"
            onClick={handleSend}
            disabled={loading || !input.trim() || !chatStatus?.available}
          >
            {loading ? '⏳' : '📤'} Send
          </button>
        </div>
        <div className="input-hint">
          Press Enter to send, Shift+Enter for new line
        </div>
      </footer>
    </div>
  )
}

export default ChatPage

