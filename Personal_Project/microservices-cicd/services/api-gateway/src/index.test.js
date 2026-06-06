const request = require('supertest');
const app = require('./src/index');

describe('API Gateway', () => {
  it('should proxy auth requests', async () => {
    const res = await request(app).get('/auth/health');
    expect(res.status).toBe(200);
  });

  it('should proxy order requests', async () => {
    const res = await request(app).get('/orders/health');
    expect(res.status).toBe(200);
  });

  it('should return gateway health', async () => {
    const res = await request(app).get('/health');
    expect(res.body.service).toBe('gateway');
    expect(res.body.status).toBe('healthy');
  });
});
