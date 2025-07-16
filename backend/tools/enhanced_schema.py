"""
tools/enhanced_schema.py
========================
Enhanced Schema Introspection for SQL Agent
Provides comprehensive database schema analysis and intelligent SQL generation
"""

import os
import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import psycopg2
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_nullable: bool
    default_value: Optional[str]
    description: Optional[str] = None
    sample_values: Optional[List[str]] = None
    constraints: Optional[List[str]] = None

@dataclass
class TableInfo:
    name: str
    columns: Dict[str, ColumnInfo]
    indexes: List[str]
    foreign_keys: List[Dict[str, str]]
    sample_data: List[Dict[str, Any]]
    row_count: int
    date_range: Optional[Dict[str, str]] = None

class SchemaIntrospector:
    """Enhanced schema introspection with intelligent SQL generation capabilities"""
    
    def __init__(self):
        self.db_config = {
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASS'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }
        self.llm = ChatOllama(model=os.getenv("LLM_MODEL", "gemma:2b"), temperature=0)
        self._schema_cache = {}
    
    def get_connection(self):
        return psycopg2.connect(**self.db_config)
    
    def introspect_table(self, table_name: str) -> TableInfo:
        """Comprehensive table introspection"""
        if table_name in self._schema_cache:
            return self._schema_cache[table_name]
        
        try:
            with self.get_connection() as conn, conn.cursor() as cur:
                # Get column information
                cur.execute("""
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        character_maximum_length,
                        numeric_precision,
                        numeric_scale
                    FROM information_schema.columns 
                    WHERE table_name = %s 
                    ORDER BY ordinal_position
                """, (table_name,))
                
                columns = {}
                for row in cur.fetchall():
                    col_name, data_type, nullable, default, char_len, num_prec, num_scale = row
                    
                    # Enhanced type info
                    if char_len:
                        data_type += f"({char_len})"
                    elif num_prec and num_scale:
                        data_type += f"({num_prec},{num_scale})"
                    elif num_prec:
                        data_type += f"({num_prec})"
                    
                    columns[col_name] = ColumnInfo(
                        name=col_name,
                        data_type=data_type,
                        is_nullable=nullable == 'YES',
                        default_value=default
                    )
                
                # Get sample values for each column (safely)
                for col_name, col_info in columns.items():
                    try:
                        cur.execute(f"""
                            SELECT DISTINCT {col_name} 
                            FROM {table_name} 
                            WHERE {col_name} IS NOT NULL 
                            LIMIT 5
                        """)
                        col_info.sample_values = [str(row[0]) for row in cur.fetchall()]
                    except Exception as e:
                        col_info.sample_values = []
                
                # Get indexes
                try:
                    cur.execute("""
                        SELECT indexname, indexdef 
                        FROM pg_indexes 
                        WHERE tablename = %s
                    """, (table_name,))
                    indexes = [f"{row[0]}: {row[1]}" for row in cur.fetchall()]
                except Exception:
                    indexes = []
                
                # Get foreign keys
                try:
                    cur.execute("""
                        SELECT
                            kcu.column_name,
                            ccu.table_name AS foreign_table_name,
                            ccu.column_name AS foreign_column_name
                        FROM information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                            ON tc.constraint_name = kcu.constraint_name
                        JOIN information_schema.constraint_column_usage AS ccu
                            ON ccu.constraint_name = tc.constraint_name
                        WHERE constraint_type = 'FOREIGN KEY' AND tc.table_name = %s
                    """, (table_name,))
                    foreign_keys = [
                        {
                            'column': row[0],
                            'references_table': row[1],
                            'references_column': row[2]
                        }
                        for row in cur.fetchall()
                    ]
                except Exception:
                    foreign_keys = []
                
                # Get sample data
                try:
                    cur.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_rows = cur.fetchall()
                    col_names = [desc[0] for desc in cur.description]
                    sample_data = []
                    for row in sample_rows:
                        row_dict = {}
                        for i, val in enumerate(row):
                            if hasattr(val, 'isoformat'):  # datetime objects
                                row_dict[col_names[i]] = val.isoformat()
                            else:
                                row_dict[col_names[i]] = str(val)
                        sample_data.append(row_dict)
                except Exception:
                    sample_data = []
                
                # Get row count
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cur.fetchone()[0]
                except Exception:
                    row_count = 0
                
                # Get date range if date column exists
                date_range = None
                date_columns = [col for col in columns.keys() 
                              if 'date' in col.lower() or 'time' in col.lower()]
                if date_columns:
                    try:
                        date_col = date_columns[0]
                        cur.execute(f"""
                            SELECT MIN({date_col})::text, MAX({date_col})::text 
                            FROM {table_name}
                        """)
                        min_date, max_date = cur.fetchone()
                        if min_date and max_date:
                            date_range = {'min': min_date, 'max': max_date}
                    except Exception:
                        pass
        
        except Exception as e:
            # Fallback for when schema introspection fails
            print(f"Schema introspection failed for {table_name}: {e}")
            columns = {
                'ticker': ColumnInfo('ticker', 'text', False, None, sample_values=['AAPL', 'MSFT']),
                'date': ColumnInfo('date', 'date', False, None, sample_values=['2025-01-01']),
                'open': ColumnInfo('open', 'numeric', True, None, sample_values=['150.0']),
                'high': ColumnInfo('high', 'numeric', True, None, sample_values=['155.0']),
                'low': ColumnInfo('low', 'numeric', True, None, sample_values=['148.0']),
                'close': ColumnInfo('close', 'numeric', True, None, sample_values=['152.0'])
            }
            
        table_info = TableInfo(
            name=table_name,
            columns=columns,
            indexes=indexes if 'indexes' in locals() else [],
            foreign_keys=foreign_keys if 'foreign_keys' in locals() else [],
            sample_data=sample_data if 'sample_data' in locals() else [],
            row_count=row_count if 'row_count' in locals() else 0,
            date_range=date_range if 'date_range' in locals() else None
        )
        
        self._schema_cache[table_name] = table_info
        return table_info
    
    def generate_schema_prompt(self, table_name: str) -> str:
        """Generate comprehensive schema description for LLM"""
        table_info = self.introspect_table(table_name)
        
        prompt = f"""
DATABASE SCHEMA FOR {table_name.upper()}:

TABLE: {table_info.name}
ROW COUNT: {table_info.row_count:,}

COLUMNS:
"""
        
        for col_name, col_info in table_info.columns.items():
            prompt += f"  • {col_name} ({col_info.data_type})"
            if not col_info.is_nullable:
                prompt += " NOT NULL"
            if col_info.default_value:
                prompt += f" DEFAULT {col_info.default_value}"
            prompt += "\n"
            
            if col_info.sample_values:
                sample_str = ", ".join(col_info.sample_values[:3])
                prompt += f"    Sample values: {sample_str}\n"
        
        if table_info.date_range:
            prompt += f"\nDATE RANGE: {table_info.date_range['min']} to {table_info.date_range['max']}\n"
        
        if table_info.sample_data:
            prompt += f"\nSAMPLE DATA:\n"
            for i, row in enumerate(table_info.sample_data):
                prompt += f"  Row {i+1}: {row}\n"
        
        return prompt

class IntelligentSQLGenerator:
    """Enhanced SQL generation with schema awareness"""
    
    def __init__(self, introspector: SchemaIntrospector):
        self.introspector = introspector
        self.llm = introspector.llm
    
    def analyze_query_context(self, query: str, table_name: str) -> Dict[str, Any]:
        """Analyze query to understand user intent and required columns"""
        table_info = self.introspector.introspect_table(table_name)
        
        analysis_prompt = f"""
Analyze this natural language query against the database schema:

QUERY: "{query}"

{self.introspector.generate_schema_prompt(table_name)}

Determine what SQL should be generated. Respond with JSON only:
{{
    "select_columns": ["column1", "column2"],
    "where_conditions": [
        {{"column": "ticker", "operator": "=", "value": "AAPL"}},
        {{"column": "date", "operator": ">=", "value": "2025-01-01"}}
    ],
    "order_by": {{"column": "date", "direction": "DESC"}},
    "limit": 10,
    "intent": "what user wants"
}}

Rules:
- Always include 'ticker' and 'date' in select_columns for stock data
- For price queries, include relevant price columns (open, close, high, low)
- Use appropriate WHERE conditions based on query context
- Order by date DESC for recent data, ASC for ranges
- Limit to reasonable number of rows (10-50)
"""
        
        try:
            response = self.llm.invoke(analysis_prompt).content.strip()
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"Query analysis failed: {e}")
        
        # Fallback analysis based on keywords
        query_lower = query.lower()
        
        # Determine columns based on query content
        select_columns = ['ticker', 'date']
        if any(word in query_lower for word in ['open', 'opening']):
            select_columns.append('open')
        if any(word in query_lower for word in ['close', 'closing', 'price']):
            select_columns.append('close')
        if 'high' in query_lower:
            select_columns.append('high')
        if 'low' in query_lower:
            select_columns.append('low')
        
        # If no specific price mentioned, default to close
        if len(select_columns) == 2:  # Only ticker and date
            select_columns.append('close')
        
        return {
            "select_columns": select_columns,
            "where_conditions": [],
            "order_by": {"column": "date", "direction": "DESC"},
            "limit": 10,
            "intent": "stock data query"
        }
    
    def generate_sql(self, query: str, ticker: str = None, dates: List[str] = None, is_range: bool = False, table_name: str = "stock_data") -> str:
        """Generate optimized SQL based on schema analysis"""
        
        # Build enhanced query with context
        enhanced_query = query
        if ticker:
            enhanced_query += f" for ticker {ticker}"
        if dates:
            if is_range and len(dates) >= 2:
                enhanced_query += f" from {dates[0]} to {dates[1]}"
            elif dates:
                enhanced_query += f" for date {dates[0]}"
        
        context = self.analyze_query_context(enhanced_query, table_name)
        table_info = self.introspector.introspect_table(table_name)
        
        # Build SELECT clause
        select_columns = context["select_columns"]
        # Validate columns exist
        valid_columns = []
        for col in select_columns:
            if col in table_info.columns:
                valid_columns.append(col)
            elif col == '*':
                valid_columns = ['*']
                break
        
        if not valid_columns:
            valid_columns = ['ticker', 'date', 'close']
        
        select_clause = ", ".join(valid_columns)
        
        # Build WHERE clause
        where_parts = []
        
        # Add conditions from context analysis
        for condition in context["where_conditions"]:
            col = condition["column"]
            op = condition["operator"]
            val = condition["value"]
            
            # Type-aware value formatting
            if col in table_info.columns:
                col_type = table_info.columns[col].data_type.lower()
                if "text" in col_type or "char" in col_type:
                    val = f"'{val}'"
                elif "date" in col_type or "timestamp" in col_type:
                    val = f"'{val}'"
            
            where_parts.append(f"{col} {op} {val}")
        
        # Add ticker condition if provided
        if ticker and not any('ticker' in part for part in where_parts):
            where_parts.append(f"ticker = '{ticker.upper()}'")
        
        # Add date conditions if provided
        if dates and not any('date' in part for part in where_parts):
            if is_range and len(dates) >= 2:
                where_parts.append(f"date BETWEEN '{dates[0]}' AND '{dates[1]}'")
            elif dates:
                where_parts.append(f"date = '{dates[0]}'")
        
        where_clause = " AND ".join(where_parts) if where_parts else ""
        
        # Build ORDER BY clause
        order_clause = ""
        if context["order_by"]:
            col = context["order_by"]["column"]
            direction = context["order_by"]["direction"]
            if col in table_info.columns:
                order_clause = f"ORDER BY {col} {direction}"
        
        # Build LIMIT clause
        limit_clause = f"LIMIT {context['limit']}" if context.get("limit") else "LIMIT 10"
        
        # Assemble final SQL
        sql_parts = [f"SELECT {select_clause}", f"FROM {table_name}"]
        
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")
        
        if order_clause:
            sql_parts.append(order_clause)
        
        if limit_clause:
            sql_parts.append(limit_clause)
        
        return "\n".join(sql_parts)

# Global instances for caching
_introspector = None
_sql_generator = None

def get_schema_introspector():
    """Get cached schema introspector instance"""
    global _introspector
    if _introspector is None:
        _introspector = SchemaIntrospector()
    return _introspector

def get_enhanced_sql_generator():
    """Get cached SQL generator instance"""
    global _sql_generator
    if _sql_generator is None:
        introspector = get_schema_introspector()
        _sql_generator = IntelligentSQLGenerator(introspector)
    return _sql_generator

def generate_enhanced_sql(query: str, ticker: str = None, dates: List[str] = None, is_range: bool = False) -> str:
    """Main entry point for enhanced SQL generation"""
    generator = get_enhanced_sql_generator()
    return generator.generate_sql(query, ticker, dates, is_range)