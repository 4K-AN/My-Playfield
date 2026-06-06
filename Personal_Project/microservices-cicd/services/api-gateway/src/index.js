const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
require('dotenv').config();

const app = express();

app.use('/auth', createProxyMiddleware({
  target: process.env.AUTH_SERVICE_URL || 'http://localhost:3001',
  changeOrigin: true
}));

app.use('/orders', createProxyMiddleware({
  target: process.env.ORDER_SERVICE_URL || 'http://localhost:3002',
  changeOrigin: true
}));

app.get('/health', (req, res) => res.json({ service: 'gateway', status: 'healthy' }));

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`API Gateway running on port ${PORT}`));

module.exports = app;
