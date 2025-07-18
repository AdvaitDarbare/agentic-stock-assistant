# User Authentication & Conversation Persistence Setup

## 🗄️ Database Setup

### 1. Create the required tables:
```bash
# Connect to your PostgreSQL database
psql -h localhost -U postgres -d agentic_stock

# Run the schema creation script
\i schema.sql
```

### 2. Install additional Python dependencies:
```bash
pip install bcrypt pyjwt asyncpg python-multipart
```

### 3. Set environment variables:
```bash
# Add to your .env file
JWT_SECRET=your-super-secret-jwt-key-change-in-production-please
DB_HOST=localhost
DB_PORT=5432
DB_NAME=agentic_stock
DB_USER=postgres
DB_PASS=secret
```

## 🔐 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user  
- `GET /auth/me` - Get current user info

### Conversations
- `GET /conversations` - Get user's conversations
- `GET /conversations/{id}/messages` - Get conversation messages
- `POST /chat` - Send message (requires auth)

## 📝 Example Usage

### 1. Register a user:
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "testuser",
    "password": "securepassword123",
    "first_name": "Test",
    "last_name": "User"
  }'
```

### 2. Login:
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com", 
    "password": "securepassword123"
  }'
```

### 3. Chat with authentication:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "query": "What is the latest price of AAPL?"
  }'
```

### 4. Get conversation history:
```bash
curl -X GET http://localhost:8000/conversations \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## 🔄 How Conversation Persistence Works

1. **New Chat**: If no `thread_id` provided, creates new conversation
2. **Continue Chat**: If `thread_id` provided, continues existing conversation
3. **Database Storage**: All messages saved to PostgreSQL with user association
4. **LangGraph Memory**: Still uses checkpointer for workflow state during conversation
5. **Cross-Session**: Conversations persist when user logs back in

## 🏗️ Database Schema

### Tables Created:
- `users` - User accounts and authentication
- `user_sessions` - JWT token management  
- `conversations` - Chat conversation groups
- `messages` - Individual chat messages
- `user_preferences` - User settings

### Key Features:
- ✅ Password hashing with bcrypt
- ✅ JWT token authentication  
- ✅ Thread-based conversation grouping
- ✅ Message type tracking (human/ai/system)
- ✅ JSONB metadata storage
- ✅ Automatic timestamps
- ✅ User-specific conversation isolation

## 🔧 Configuration Notes

### Checkpoint Tables
The existing checkpoint tables (`checkpoints`, `checkpoint_writes`, etc.) are still used by LangGraph for workflow state management within conversations. The new tables handle user authentication and long-term conversation storage.

### Security Considerations
- Change `JWT_SECRET` in production
- Use HTTPS in production
- Configure CORS properly for your frontend domain
- Consider rate limiting for auth endpoints
- Implement password strength requirements
- Add email verification if needed

## 🚀 Production Recommendations

1. **Database**: Use connection pooling
2. **JWT**: Implement refresh tokens
3. **Security**: Add rate limiting, CORS configuration
4. **Monitoring**: Add authentication logging
5. **Backup**: Regular database backups for conversation history