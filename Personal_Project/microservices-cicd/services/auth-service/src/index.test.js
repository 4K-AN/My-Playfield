const request = require('supertest');
const app = require('./src/index');

describe('Auth Service', () => {
  it('should register a new user', async () => {
    const res = await request(app)
      .post('/register')
      .send({ email: 'test@example.com', password: 'password123' });
    expect(res.status).toBe(201);
    expect(res.body).toHaveProperty('id');
  });

  it('should login and return a token', async () => {
    await request(app).post('/register').send({ email: 'login@test.com', password: 'pass123' });
    const res = await request(app)
      .post('/login')
      .send({ email: 'login@test.com', password: 'pass123' });
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('token');
  });

  it('should reject invalid login', async () => {
    const res = await request(app)
      .post('/login')
      .send({ email: 'wrong@test.com', password: 'wrong' });
    expect(res.status).toBe(401);
  });

  it('should return health status', async () => {
    const res = await request(app).get('/health');
    expect(res.body.status).toBe('healthy');
    expect(res.body.service).toBe('auth');
  });
});
