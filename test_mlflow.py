import os
from dotenv import load_dotenv
import mlflow
import mlflow.tracking

# Load environment variables
load_dotenv()

print("Testing MLflow Connection...")
print("=" * 50)

# Check environment variables
required_vars = ["DATABRICKS_TOKEN", "DATABRICKS_HOST", "MLFLOW_TRACKING_URI"]
missing_vars = []

for var in required_vars:
    value = os.getenv(var)
    if value:
        if "TOKEN" in var:
            print(f"✅ {var}: {'*' * 10}...{value[-4:]}")
        else:
            print(f"✅ {var}: {value}")
    else:
        print(f"❌ {var}: NOT SET")
        missing_vars.append(var)

if missing_vars:
    print(f"\n❌ Missing required environment variables: {missing_vars}")
    print("Please set these in your .env file")
    exit(1)

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
print(f"\nMLflow Tracking URI: {mlflow.get_tracking_uri()}")

# Test basic MLflow connection
try:
    # List experiments to test connection
    experiments = mlflow.search_experiments()
    print(f"✅ Successfully connected to MLflow")
    print(f"✅ Found {len(experiments)} experiments")
    
    # Test specific experiment if provided
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "3507954515130955")
    
    try:
        experiment = mlflow.get_experiment(experiment_id)
        if experiment:
            print(f"✅ Experiment found: '{experiment.name}' (ID: {experiment_id})")
            
            # Test creating a run
            mlflow.set_experiment(experiment_id=experiment_id)
            
            with mlflow.start_run() as run:
                # Log some test data
                mlflow.log_param("test_type", "connection_test")
                mlflow.log_param("timestamp", str(mlflow.utils.time.get_current_time_millis()))
                mlflow.log_metric("connection_success", 1.0)
                
                print(f"✅ Created test run: {run.info.run_id}")
                print(f"✅ Run URL: {run.info.artifact_uri}")
                
            print(f"\n🎉 MLflow connection fully working!")
            print(f"🔗 Check your experiment at: {os.getenv('DATABRICKS_HOST')}/ml/experiments/{experiment_id}")
            
        else:
            print(f"❌ Experiment {experiment_id} not found")
            
    except Exception as exp_error:
        print(f"❌ Experiment access failed: {exp_error}")
        print("Available experiments:")
        for exp in experiments[:5]:  # Show first 5
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
        
except Exception as e:
    print(f"❌ MLflow connection failed: {e}")
    print("\nTroubleshooting steps:")
    print("1. Verify DATABRICKS_TOKEN is valid and not expired")
    print("2. Check DATABRICKS_HOST format (should be https://dbc-xxx.cloud.databricks.com)")
    print("3. Ensure MLFLOW_TRACKING_URI is set to 'databricks'")
    print("4. Verify network connectivity to Databricks")
    print("5. Check if your token has MLflow permissions")
    
    # Additional debugging info
    print(f"\nDebugging info:")
    print(f"Python MLflow version: {mlflow.__version__}")
    print(f"Current working directory: {os.getcwd()}")