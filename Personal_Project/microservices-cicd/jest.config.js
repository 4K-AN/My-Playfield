module.exports = {
  testEnvironment: 'node',
  coverageDirectory: 'coverage',
  collectCoverage: true,
  collectCoverageFrom: ['src/**/*.js'],
  testMatch: ['**/__tests__/**/*.js', '**/*.test.js'],
  verbose: true
};
