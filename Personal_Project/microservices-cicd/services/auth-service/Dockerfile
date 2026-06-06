FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --production=false

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build --if-present
RUN npm run lint && npm test -- --passWithNoTests

FROM node:20-alpine AS runner
WORKDIR /app
COPY package*.json ./
COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/src ./src
ENV NODE_ENV=production
USER node
EXPOSE ${PORT:-3001}
CMD ["npm", "start"]
