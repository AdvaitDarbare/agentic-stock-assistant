# FinanceScope - AI Stock & Market Analysis

Intelligent financial analysis with real-time stock data, news insights, and sentiment analysis powered by multi-agent LangGraph workflows.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js and npm
- Python 3.11+

### Complete Startup (Recommended for Development)
```bash
# 1. Start Docker services (database + MLFlow)
cd backend
docker compose up db mlflow -d

# 2. Start MCP servers (new terminal)
cd backend
python3 start_all_servers.py

# 3. Start backend locally (new terminal)
cd backend
uvicorn main:app --reload

# 4. Start frontend (new terminal)
cd frontend
npm run dev
```

### Alternative: Full Docker Setup
```bash
# 1. Start all Docker services
cd backend
docker compose up -d

# 2. Start MCP servers (new terminal)
cd backend
python3 start_all_servers.py

# 3. Start frontend (new terminal)
cd frontend
npm run dev
```

## 🌐 Access URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLFlow UI**: http://localhost:5001
- **Database**: localhost:5432 (internal)

## 📊 MLFlow Integration

### What is MLFlow?
MLFlow is an open-source platform for managing machine learning experiments, tracking metrics, and versioning models. In FinanceScope, it tracks:

- **Agent Performance**: Response times, success rates
- **Model Metrics**: LLM token usage, accuracy metrics
- **Experiment Tracking**: A/B testing of different prompts
- **Parameter Logging**: Model parameters, configurations

### MLFlow Features in FinanceScope:
- **Experiment Tracking**: All agent interactions are logged
- **Model Versioning**: Track different model configurations
- **Metrics Dashboard**: Visualize performance over time
- **Parameter Comparison**: Compare different agent settings

### Test MLFlow:
```bash
# Test the MLFlow connection
cd backend
python3 tests/test_mlflow.py
```

## 🐘 Database Access

### PostgreSQL Shell Access
```bash
# Interactive PostgreSQL shell
docker exec -it backend-db-1 psql -U postgres -d agentic_stock
```

### Database Connection Details
- **Host**: localhost
- **Port**: 5432
- **Database**: agentic_stock
- **Username**: postgres
- **Password**: secret

### Common PostgreSQL Commands
```sql
-- See all tables
\dt

-- Look at stock_data table structure
\d stock_data

-- See sample data
SELECT * FROM stock_data LIMIT 5;

-- Count total rows
SELECT COUNT(*) FROM stock_data;

-- Exit PostgreSQL shell
\q
```

### One-time Database Queries (without entering shell)
```bash
# Check database version
docker exec backend-db-1 psql -U postgres -d agentic_stock -c "SELECT version();"

# List all tables
docker exec backend-db-1 psql -U postgres -d agentic_stock -c "\dt"

# Count rows in stock_data table
docker exec backend-db-1 psql -U postgres -d agentic_stock -c "SELECT COUNT(*) FROM stock_data;"
```

## 🛠️ Development Commands

### Check Container Status
```bash
docker ps
```

### View Service Logs
```bash
# Backend logs (if using Docker)
docker logs backend-api-1 -f

# MLFlow logs
docker logs backend-mlflow-1 -f

# Database logs
docker logs backend-db-1 -f
```

### Restart Services
```bash
# Restart Docker services
docker compose down
docker compose up -d

# Restart MCP servers
# Ctrl+C to stop, then run again:
python3 start_all_servers.py
```

## 🔧 MCP Server URLs

When MCP servers are running:
- **SQL Agent**: http://localhost:8010/mcp
- **News Agent**: http://localhost:8020/mcp
- **Fallback Agent**: http://localhost:8030/mcp
- **Sentiment Agent**: http://localhost:8040/mcp

## 📊 Features

- **Stock Price Analysis**: Real-time stock data queries and analysis
- **News Integration**: Latest financial news and headlines
- **Sentiment Analysis**: Correlation between news sentiment and stock prices
- **Multi-Agent System**: Specialized agents for different data sources
- **Persistent Conversations**: Chat history maintained across sessions
- **Interactive UI**: Modern chat interface with conversation management
- **MLFlow Tracking**: Experiment tracking and model performance monitoring

## 🛑 Stopping the Application

```bash
# Stop MCP servers
# Ctrl+C in the terminal running start_all_servers.py

# Stop frontend
# Ctrl+C in the terminal running npm run dev

# Stop backend (if running locally)
# Ctrl+C in the terminal running uvicorn

# Stop Docker services
cd backend
docker compose down
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Check what's using ports
lsof -i :5432  # Database
lsof -i :5001  # MLFlow
lsof -i :8000  # Backend

# Kill existing containers
docker stop backend-db-1 backend-api-1 backend-mlflow-1
docker rm backend-db-1 backend-api-1 backend-mlflow-1
```

### Database Connection Issues
```bash
# Check if database is running
docker ps | grep db

# Test database connection
docker exec backend-db-1 psql -U postgres -d agentic_stock -c "SELECT current_database();"
```

### MLFlow Issues
```bash
# Check MLFlow service
curl http://localhost:5001/health

# View MLFlow logs
docker logs backend-mlflow-1 -f

# Test MLFlow connection
python3 tests/test_mlflow.py
```

### MCP Servers Not Starting
```bash
# Check for Python import errors
python3 -c "from graph import run_query_with_persistence; print('Import successful')"

# Check if ports are available
netstat -an | grep -E "(8010|8020|8030|8040)"
```

## 📁 Project Structure

```
.
├── backend/
│   ├── agents/              # MCP server agents
│   ├── tools/               # Database and utility tools
│   ├── tests/               # Test files and ticker mapping
│   ├── docker-compose.yml   # Docker services configuration
│   ├── main.py             # FastAPI application
│   └── graph.py            # LangGraph workflow
├── frontend/
│   ├── src/
│   │   ├── app/            # Next.js app router
│   │   └── components/     # React components
│   └── package.json
└── README.md
```

## 🔄 Development Workflow

1. **Start Database & MLFlow**: `docker compose up db mlflow -d`
2. **Start MCP Servers**: `python3 start_all_servers.py`
3. **Start Backend**: `uvicorn main:app --reload` (for development)
4. **Start Frontend**: `npm run dev`
5. **Access Application**: http://localhost:3000
6. **Monitor with MLFlow**: http://localhost:5001

## 🎯 Example Queries

Try these queries in the application:
- "What's the latest news on MSFT?"
- "Show me AAPL stock prices from 2025-06-06 to 2025-06-11"
- "Analyze sentiment for TSLA news"
- "What's the correlation between NVDA news and stock price?"

Happy coding! 🎉