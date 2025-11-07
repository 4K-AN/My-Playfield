import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, ReferenceDot } from 'recharts';

const CloudStorageGraph = () => {
  // Generate data points for t from 0 to 5 hours
  const generateData = () => {
    const data = [];
    for (let t = 0; t <= 5; t += 0.1) {
      // g(t) = 2t² + t
      const g_t = 2 * t * t + t;
      // f(x) = 3x + 1, so D(t) = f(g(t)) = 3(2t² + t) + 1 = 6t² + 3t + 1
      const D_t = 6 * t * t + 3 * t + 1;
      // D'(t) = 12t + 3
      const D_prime_t = 12 * t + 3;
      
      data.push({
        t: parseFloat(t.toFixed(1)),
        'D(t)': parseFloat(D_t.toFixed(2)),
        "D'(t)": parseFloat(D_prime_t.toFixed(2))
      });
    }
    return data;
  };

  const data = generateData();
  
  // Find the point at t = 3
  const pointAtT3 = data.find(point => point.t === 3.0);

  return (
    <div className="w-full p-6 bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-center text-indigo-800 mb-2">
          Cloud Storage Data Occupancy Analysis
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Grafik Fungsi D(t) dan Turunannya D'(t)
        </p>

        {/* Main Graph */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">
            Data Occupancy D(t) dan Velocity D'(t) vs Time
          </h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
              <XAxis 
                dataKey="t" 
                label={{ value: 'Time (hours)', position: 'insideBottom', offset: -10 }}
                stroke="#6b7280"
              />
              <YAxis 
                label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
                stroke="#6b7280"
              />
              <Tooltip 
                formatter={(value, name) => [
                  `${value} ${name === 'D(t)' ? 'GB' : 'GB/hour'}`, 
                  name
                ]}
                labelFormatter={(t) => `Time: ${t} hours`}
                contentStyle={{
                  backgroundColor: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px'
                }}
              />
              <Legend />
              
              {/* Reference line at t = 3 */}
              <ReferenceLine x={3} stroke="#ef4444" strokeDasharray="5 5" />
              
              {/* Data Occupancy D(t) */}
              <Line 
                type="monotone" 
                dataKey="D(t)" 
                stroke="#3b82f6" 
                strokeWidth={3}
                dot={false}
                name="D(t) - Data Occupancy"
              />
              
              {/* Velocity D'(t) */}
              <Line 
                type="monotone" 
                dataKey="D'(t)" 
                stroke="#10b981" 
                strokeWidth={3}
                dot={false}
                name="D'(t) - Occupancy Velocity"
              />
              
              {/* Highlight point at t = 3 */}
              {pointAtT3 && (
                <>
                  <ReferenceDot x={3} y={pointAtT3['D(t)']} fill="#3b82f6" r={6} />
                  <ReferenceDot x={3} y={pointAtT3["D'(t)"]} fill="#10b981" r={6} />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Analysis Cards */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          {/* Function Definitions */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-indigo-700">Function Definitions</h3>
            <div className="space-y-3">
              <div className="bg-blue-50 p-3 rounded-lg">
                <p className="font-mono text-sm"><strong>g(t)</strong> = 2t² + t</p>
              </div>
              <div className="bg-blue-50 p-3 rounded-lg">
                <p className="font-mono text-sm"><strong>f(x)</strong> = 3x + 1</p>
              </div>
              <div className="bg-indigo-50 p-3 rounded-lg">
                <p className="font-mono text-sm"><strong>D(t)</strong> = f(g(t)) = 6t² + 3t + 1</p>
              </div>
              <div className="bg-green-50 p-3 rounded-lg">
                <p className="font-mono text-sm"><strong>D'(t)</strong> = 12t + 3</p>
              </div>
            </div>
          </div>

          {/* Key Values */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-indigo-700">Values at t = 3 hours</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center bg-blue-50 p-3 rounded-lg">
                <span className="font-semibold">Data Occupancy:</span>
                <span className="font-mono text-blue-600">{pointAtT3?.['D(t)']} GB</span>
              </div>
              <div className="flex justify-between items-center bg-green-50 p-3 rounded-lg">
                <span className="font-semibold">Occupancy Velocity:</span>
                <span className="font-mono text-green-600">{pointAtT3?.["D'(t)"]} GB/hour</span>
              </div>
              <div className="bg-yellow-50 p-3 rounded-lg">
                <p className="text-sm text-gray-700">
                  <strong>Interpretation:</strong> At hour 3, the system stores {pointAtT3?.['D(t)']} GB 
                  and is growing at {pointAtT3?.["D'(t)"]} GB/hour.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Chain Rule Explanation */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4 text-indigo-700">Chain Rule Application</h3>
          <div className="bg-gray-50 p-4 rounded-lg font-mono text-sm">
            <div className="mb-2"><strong>D'(t) = f'(g(t)) × g'(t)</strong></div>
            <div className="mb-2">f'(x) = 3</div>
            <div className="mb-2">g'(t) = 4t + 1</div>
            <div className="mb-2">D'(t) = 3 × (4t + 1) = 12t + 3</div>
            <div className="text-green-600"><strong>D'(3) = 12(3) + 3 = 39 GB/hour</strong></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CloudStorageGraph;