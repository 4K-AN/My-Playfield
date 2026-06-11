import { useState } from 'react'
import axios from 'axios'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts'
import './App.css'

function App() {
  const [formData, setFormData] = useState({
    YearsExperience: 5,
    Education: 'Bachelor',
    Role: 'Software Engineer',
    Location: 'New York'
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: name === 'YearsExperience' ? parseFloat(value) : value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    
    try {
      const response = await axios.post('http://localhost:8000/predict', formData)
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while predicting.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="glass-panel main-panel">
        <header>
          <h1>Salary Predictor</h1>
          <p>Powered by Machine Learning & SHAP</p>
        </header>

        <div className="content">
          <form onSubmit={handleSubmit} className="prediction-form">
            <div className="form-group">
              <label>Years of Experience</label>
              <input
                type="number"
                name="YearsExperience"
                value={formData.YearsExperience}
                onChange={handleChange}
                min="0"
                step="0.5"
                required
              />
            </div>

            <div className="form-group">
              <label>Education</label>
              <select name="Education" value={formData.Education} onChange={handleChange}>
                <option value="High School">High School</option>
                <option value="Bachelor">Bachelor</option>
                <option value="Master">Master</option>
                <option value="PhD">PhD</option>
              </select>
            </div>

            <div className="form-group">
              <label>Role</label>
              <select name="Role" value={formData.Role} onChange={handleChange}>
                <option value="Software Engineer">Software Engineer</option>
                <option value="Data Scientist">Data Scientist</option>
                <option value="Product Manager">Product Manager</option>
                <option value="Designer">Designer</option>
                <option value="DevOps">DevOps</option>
              </select>
            </div>

            <div className="form-group">
              <label>Location</label>
              <select name="Location" value={formData.Location} onChange={handleChange}>
                <option value="New York">New York</option>
                <option value="San Francisco">San Francisco</option>
                <option value="Remote">Remote</option>
                <option value="London">London</option>
                <option value="Berlin">Berlin</option>
              </select>
            </div>

            <button type="submit" disabled={loading} className="submit-btn">
              {loading ? 'Predicting...' : 'Predict Salary'}
            </button>
          </form>

          {error && <div className="error-message">{error}</div>}

          {result && (
            <div className="results-panel slide-in">
              <div className="prediction-box">
                <h2>Estimated Salary</h2>
                <div className="salary-value">
                  ${result.prediction.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </div>
                <div className="base-value">
                  Base Average: ${result.base_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </div>
              </div>

              <div className="shap-box">
                <h3>Feature Impact (SHAP Values)</h3>
                <p className="shap-desc">How each feature contributed to this specific prediction compared to the base average.</p>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={result.shap_values}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 12 }} />
                      <YAxis dataKey="feature" type="category" tick={{ fill: '#6b7280', fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#fff', border: '1px solid #d1d5db', borderRadius: 0, color: '#111' }}
                        formatter={(value) => [`$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, 'Impact']}
                      />
                      <Bar dataKey="value" radius={[0, 0, 0, 0]}>
                        {
                          result.shap_values.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#22c55e' : '#ef4444'} />
                          ))
                        }
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
