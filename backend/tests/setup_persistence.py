#!/usr/bin/env python3
"""
Setup script for LangGraph persistence tables (PostgreSQL version)
Note: Currently using memory checkpointer, this script is for future PostgreSQL persistence
"""

import asyncio
import os
from dotenv import load_dotenv
import asyncpg

load_dotenv()

async def setup_persistence_tables():
    """Create the necessary tables for LangGraph persistence"""
    
    print("ℹ️  Setting up PostgreSQL checkpointer for LangGraph persistence")
    print("💾 Conversations will persist across server restarts")
    print("🔄 Creating necessary tables for PostgreSQL persistence")
    
    # Database connection parameters
    conn_params = {
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT')),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASS')
    }
    
    print("🔗 Testing PostgreSQL connection...")
    
    try:
        conn = await asyncpg.connect(**conn_params)
        
        # Create schema for future LangGraph checkpoints
        print("📝 Creating langgraph_checkpoints schema (for future use)...")
        await conn.execute("""
            CREATE SCHEMA IF NOT EXISTS langgraph_checkpoints;
        """)
        
        # Create the checkpoints table (for future use)
        print("📋 Creating checkpoints table (for future use)...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS langgraph_checkpoints.checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                checkpoint JSONB NOT NULL,
                metadata JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_id)
            );
        """)
        
        # Create index for better performance
        print("🔍 Creating indexes...")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id 
            ON langgraph_checkpoints.checkpoints(thread_id);
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at 
            ON langgraph_checkpoints.checkpoints(created_at);
        """)
        
        # Create the channel_values table (for LangGraph state)
        print("🗂️  Creating channel_values table (for future use)...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS langgraph_checkpoints.channel_values (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                value JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_id, channel)
            );
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_channel_values_thread_id 
            ON langgraph_checkpoints.channel_values(thread_id);
        """)
        
        await conn.close()
        
        print("✅ Database tables created successfully!")
        print("📊 Your PostgreSQL database is ready for LangGraph persistence")
        print("🔄 LangGraph checkpointer will now use PostgreSQL for persistence")
        
    except Exception as e:
        print(f"❌ Error setting up persistence tables: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(setup_persistence_tables())