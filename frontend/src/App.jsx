import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import SearchPage from './SearchPage'
import CaseDetail from './CaseDetail'
import ChatPage from './ChatPage'
import StrategyPage from './StrategyPage'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/strategy" element={<StrategyPage />} />
        <Route path="/case/:caseIndex" element={<CaseDetail />} />
      </Routes>
    </Router>
  )
}

export default App
