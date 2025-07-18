"""
Authentication and User Management System
"""

import os
import jwt
import bcrypt
import asyncpg
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-jwt-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days

# Database connection
async def get_db_connection():
    """Get database connection"""
    connection = await asyncpg.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'agentic_stock'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASS', 'secret')
    )
    return connection

# Pydantic models
class UserRegistration(BaseModel):
    email: EmailStr
    username: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class User(BaseModel):
    id: int
    email: str
    username: str
    first_name: Optional[str]
    last_name: Optional[str]
    is_active: bool
    created_at: datetime

# Password utilities
def hash_password(password: str) -> str:
    """Hash a password with bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# JWT utilities
def create_access_token(user_id: int, email: str) -> str:
    """Create a JWT access token"""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(tz=timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Authentication dependency
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get the current authenticated user"""
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    # Get user from database
    conn = await get_db_connection()
    try:
        user_record = await conn.fetchrow(
            "SELECT id, email, username, first_name, last_name, is_active, created_at FROM users WHERE id = $1",
            user_id
        )
        
        if not user_record:
            raise HTTPException(status_code=401, detail="User not found")
        
        if not user_record['is_active']:
            raise HTTPException(status_code=401, detail="User account is inactive")
        
        return User(**dict(user_record))
    
    finally:
        await conn.close()

# Authentication functions
async def register_user(user_data: UserRegistration) -> User:
    """Register a new user"""
    conn = await get_db_connection()
    try:
        # Check if user already exists
        existing_user = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1 OR username = $2",
            user_data.email, user_data.username
        )
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email or username already exists")
        
        # Hash password
        password_hash = hash_password(user_data.password)
        
        # Insert new user
        user_record = await conn.fetchrow(
            """
            INSERT INTO users (email, username, password_hash, first_name, last_name)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, email, username, first_name, last_name, is_active, created_at
            """,
            user_data.email, user_data.username, password_hash,
            user_data.first_name, user_data.last_name
        )
        
        return User(**dict(user_record))
    
    finally:
        await conn.close()

async def authenticate_user(login_data: UserLogin) -> tuple[User, str]:
    """Authenticate user and return user data + JWT token"""
    conn = await get_db_connection()
    try:
        # Get user by email
        user_record = await conn.fetchrow(
            "SELECT id, email, username, password_hash, first_name, last_name, is_active, created_at FROM users WHERE email = $1",
            login_data.email
        )
        
        if not user_record:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        if not user_record['is_active']:
            raise HTTPException(status_code=401, detail="User account is inactive")
        
        # Verify password
        if not verify_password(login_data.password, user_record['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Update last login
        await conn.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = $1",
            user_record['id']
        )
        
        # Create user object (without password_hash)
        user_dict = dict(user_record)
        del user_dict['password_hash']
        user = User(**user_dict)
        
        # Create access token
        token = create_access_token(user.id, user.email)
        
        return user, token
    
    finally:
        await conn.close()

# Conversation management
async def create_conversation(user_id: int, title: str, thread_id: str) -> int:
    """Create a new conversation for a user"""
    conn = await get_db_connection()
    try:
        conversation_id = await conn.fetchval(
            """
            INSERT INTO conversations (user_id, title, thread_id)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            user_id, title, thread_id
        )
        return conversation_id
    finally:
        await conn.close()

async def get_user_conversations(user_id: int, limit: int = 50):
    """Get all conversations for a user"""
    conn = await get_db_connection()
    try:
        conversations = await conn.fetch(
            """
            SELECT c.id, c.title, c.thread_id, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count,
                   MAX(m.created_at) as last_message_at
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.user_id = $1 AND c.is_archived = FALSE
            GROUP BY c.id, c.title, c.thread_id, c.created_at, c.updated_at
            ORDER BY COALESCE(MAX(m.created_at), c.created_at) DESC
            LIMIT $2
            """,
            user_id, limit
        )
        return [dict(conv) for conv in conversations]
    finally:
        await conn.close()

async def get_conversation_by_thread_id(user_id: int, thread_id: str):
    """Get a conversation by thread_id for a specific user"""
    conn = await get_db_connection()
    try:
        conversation = await conn.fetchrow(
            "SELECT * FROM conversations WHERE user_id = $1 AND thread_id = $2",
            user_id, thread_id
        )
        return dict(conversation) if conversation else None
    finally:
        await conn.close()

async def save_message(conversation_id: int, message_type: str, content: str, metadata: dict = None):
    """Save a message to a conversation"""
    import json
    conn = await get_db_connection()
    try:
        # Convert metadata dict to JSON string for JSONB storage
        metadata_json = json.dumps(metadata or {})
        
        message_id = await conn.fetchval(
            """
            INSERT INTO messages (conversation_id, message_type, content, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING id
            """,
            conversation_id, message_type, content, metadata_json
        )
        
        # Update conversation timestamp
        await conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
            conversation_id
        )
        
        return message_id
    finally:
        await conn.close()

async def get_conversation_messages(conversation_id: int, limit: int = 100):
    """Get all messages for a conversation"""
    conn = await get_db_connection()
    try:
        messages = await conn.fetch(
            """
            SELECT id, message_type, content, metadata, created_at
            FROM messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            LIMIT $2
            """,
            conversation_id, limit
        )
        return [dict(msg) for msg in messages]
    finally:
        await conn.close()