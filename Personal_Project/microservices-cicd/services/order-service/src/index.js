const express = require('express');
const mongoose = require('mongoose');
require('dotenv').config();

const app = express();
app.use(express.json());

mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/order-service');

const OrderSchema = new mongoose.Schema({
  userId: { type: String, required: true },
  items: [{ name: String, price: Number, quantity: Number }],
  total: { type: Number, required: true },
  status: { type: String, default: 'pending' }
});
const Order = mongoose.model('Order', OrderSchema);

app.get('/orders/:userId', async (req, res) => {
  const orders = await Order.find({ userId: req.params.userId });
  res.json(orders);
});

app.post('/orders', async (req, res) => {
  const order = await Order.create(req.body);
  res.status(201).json(order);
});

app.patch('/orders/:id/cancel', async (req, res) => {
  const order = await Order.findByIdAndUpdate(req.params.id, { status: 'cancelled' }, { new: true });
  res.json(order);
});

app.get('/health', (req, res) => res.json({ service: 'orders', status: 'healthy' }));

const PORT = process.env.PORT || 3002;
app.listen(PORT, () => console.log(`Order service running on port ${PORT}`));

module.exports = app;
