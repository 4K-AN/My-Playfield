const request = require('supertest');
const app = require('./src/index');

describe('Order Service', () => {
  it('should create an order', async () => {
    const res = await request(app)
      .post('/orders')
      .send({ userId: 'user123', items: [{ name: 'Item', price: 10, quantity: 2 }], total: 20 });
    expect(res.status).toBe(201);
    expect(res.body.status).toBe('pending');
  });

  it('should get orders for a user', async () => {
    const res = await request(app).get('/orders/user123');
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body)).toBe(true);
  });

  it('should cancel an order', async () => {
    const order = await request(app).post('/orders').send({
      userId: 'user456', items: [{ name: 'Item', price: 10, quantity: 1 }], total: 10
    });
    const res = await request(app).patch(`/orders/${order.body._id}/cancel`);
    expect(res.body.status).toBe('cancelled');
  });

  it('should return health status', async () => {
    const res = await request(app).get('/health');
    expect(res.body.status).toBe('healthy');
  });
});
