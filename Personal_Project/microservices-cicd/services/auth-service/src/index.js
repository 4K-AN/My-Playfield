const express = require('express');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const mongoose = require('mongoose');
require('dotenv').config();

const app = express();
app.use(express.json());

mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/auth-service');

const UserSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true }
});
const User = mongoose.model('User', UserSchema);

const authenticate = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Unauthorized' });
  jwt.verify(token, process.env.JWT_SECRET || 'secret', (err, user) => {
    if (err) return res.status(403).json({ error: 'Forbidden' });
    req.user = user;
    next();
  });
};

app.post('/register', async (req, res) => {
  try {
    const hashedPassword = await bcrypt.hash(req.body.password, 10);
    const user = await User.create({ email: req.body.email, password: hashedPassword });
    res.status(201).json({ id: user._id, email: user.email });
  } catch (err) {
    res.status(400).json({ error: 'User already exists' });
  }
});

app.post('/login', async (req, res) => {
  const user = await User.findOne({ email: req.body.email });
  if (!user || !await bcrypt.compare(req.body.password, user.password)) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }
  const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET || 'secret', { expiresIn: '1h' });
  res.json({ token });
});

app.get('/verify', authenticate, (req, res) => {
  res.json({ user: req.user });
});

app.get('/health', (req, res) => res.json({ service: 'auth', status: 'healthy' }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Auth service running on port ${PORT}`));

module.exports = app;
