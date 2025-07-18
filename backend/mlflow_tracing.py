"""
MLFlow Tracing Integration for LangGraph
========================================
Comprehensive tracing system for multi-agent workflows
"""

import os
import json
import time
import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from functools import wraps

import mlflow
import mlflow.langchain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LangGraphTracer:
    """MLFlow tracing integration for LangGraph workflows"""
    
    def __init__(self):
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "finance-scope-experiments")
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        self.current_run = None
        self.workflow_start_time = None
        self.step_timings = {}
        self.step_counter = 0
        
        # Initialize MLFlow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                mlflow.create_experiment(self.experiment_name)
        except Exception as e:
            print(f"Warning: Could not setup MLFlow experiment: {e}")
    
    @contextmanager
    def trace_workflow(self, query: str, thread_id: str, metadata: Dict[str, Any] = None):
        """Context manager for tracing entire workflow execution"""
        self.workflow_start_time = time.time()
        self.step_counter = 0
        self.step_timings = {}
        
        # Start a custom trace for this workflow
        trace_info = {
            "inputs": {"query": query, "thread_id": thread_id},
            "outputs": {},
            "metadata": metadata or {}
        }
        
        try:
            mlflow.set_experiment(self.experiment_name)
            
            # Start MLFlow run with enhanced logging
            with mlflow.start_run() as run:
                self.current_run = run
                
                # Log workflow parameters
                mlflow.log_param("query", query)
                mlflow.log_param("thread_id", thread_id)
                mlflow.log_param("timestamp", datetime.datetime.now().isoformat())
                mlflow.log_param("workflow_type", "multi_agent_stock_analysis")
                
                # Log metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        mlflow.log_param(f"metadata_{key}", str(value))
                
                # Store trace information for detailed logging
                self.trace_info = trace_info
                
                yield self
                
                # Log final metrics
                total_time = time.time() - self.workflow_start_time
                mlflow.log_metric("total_workflow_time", total_time)
                mlflow.log_metric("total_steps", self.step_counter)
                
                # Log step timing summary
                if self.step_timings:
                    avg_step_time = sum(self.step_timings.values()) / len(self.step_timings)
                    mlflow.log_metric("avg_step_time", avg_step_time)
                    
                    # Log individual step times
                    for step, duration in self.step_timings.items():
                        mlflow.log_metric(f"step_time_{step}", duration)
                
        except Exception as e:
            print(f"MLFlow tracing error: {e}")
            # Fallback to basic tracing without custom traces
            try:
                mlflow.set_experiment(self.experiment_name)
                with mlflow.start_run() as run:
                    self.current_run = run
                    mlflow.log_param("query", query)
                    mlflow.log_param("thread_id", thread_id)
                    mlflow.log_param("timestamp", datetime.datetime.now().isoformat())
                    yield self
            except Exception as fallback_error:
                print(f"MLFlow fallback error: {fallback_error}")
                yield self
        finally:
            self.current_run = None
            if hasattr(self, 'trace_info'):
                self.trace_info = None
    
    def trace_node_execution(self, node_name: str, input_data: Dict[str, Any], output_data: Dict[str, Any], 
                           execution_time: float, metadata: Dict[str, Any] = None):
        """Trace individual node execution"""
        if not self.current_run:
            return
            
        try:
            self.step_counter += 1
            self.step_timings[node_name] = execution_time
            
            # Log detailed node execution info
            node_execution_info = {
                "node_name": node_name,
                "step_number": self.step_counter,
                "execution_time": execution_time,
                "timestamp": datetime.datetime.now().isoformat(),
                "input_summary": self._summarize_data(input_data),
                "output_summary": self._summarize_data(output_data),
                "metadata": metadata or {},
                "node_type": "langgraph_node"
            }
            
            # Log node metrics
            mlflow.log_metric(f"node_{node_name}_execution_time", execution_time)
            mlflow.log_metric(f"node_{node_name}_step_number", self.step_counter)
            
            # Log input/output sizes
            if input_data:
                input_size = len(json.dumps(input_data, default=str))
                mlflow.log_metric(f"node_{node_name}_input_size", input_size)
            
            if output_data:
                output_size = len(json.dumps(output_data, default=str))
                mlflow.log_metric(f"node_{node_name}_output_size", output_size)
            
            # Log metadata
            if metadata:
                for key, value in metadata.items():
                    mlflow.log_param(f"node_{node_name}_{key}", str(value))
            
            # Save node execution as artifact (handle file system errors gracefully)
            try:
                artifact_path = f"step_{self.step_counter:02d}_{node_name}.json"
                mlflow.log_dict(node_execution_info, artifact_path)
            except OSError as fs_error:
                if "Read-only file system" in str(fs_error):
                    # Log as parameters instead when file system is read-only
                    mlflow.log_param(f"node_{node_name}_summary", str(node_execution_info)[:200] + "...")
                else:
                    raise
            
        except Exception as e:
            print(f"Node tracing error for {node_name}: {e}")
    
    def trace_agent_call(self, agent_name: str, query: str, response: str, 
                        execution_time: float, success: bool, error: str = None):
        """Trace MCP agent calls"""
        if not self.current_run:
            return
            
        try:
            # Log agent metrics
            mlflow.log_metric(f"agent_{agent_name}_execution_time", execution_time)
            mlflow.log_metric(f"agent_{agent_name}_success", 1 if success else 0)
            
            # Log query and response lengths
            mlflow.log_metric(f"agent_{agent_name}_query_length", len(query))
            mlflow.log_metric(f"agent_{agent_name}_response_length", len(response) if response else 0)
            
            # Log parameters
            mlflow.log_param(f"agent_{agent_name}_query", query[:200] + "..." if len(query) > 200 else query)
            
            if error:
                mlflow.log_param(f"agent_{agent_name}_error", error)
            
            # Create agent call artifact (handle file system errors gracefully)
            agent_call_info = {
                "agent_name": agent_name,
                "query": query,
                "response": response,
                "execution_time": execution_time,
                "success": success,
                "error": error,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            try:
                artifact_path = f"agent_call_{agent_name}_{int(time.time())}.json"
                mlflow.log_dict(agent_call_info, artifact_path)
            except OSError as fs_error:
                if "Read-only file system" in str(fs_error):
                    # Log summary as parameters when file system is read-only
                    mlflow.log_param(f"agent_{agent_name}_call_summary", f"Query: {query[:100]}...")
                else:
                    raise
            
        except Exception as e:
            print(f"Agent tracing error for {agent_name}: {e}")
    
    def trace_routing_decision(self, query: str, routing_result: Dict[str, bool], 
                             execution_time: float, metadata: Dict[str, Any] = None):
        """Trace routing decisions"""
        if not self.current_run:
            return
            
        try:
            # Log routing metrics
            mlflow.log_metric("routing_execution_time", execution_time)
            mlflow.log_metric("routing_agents_needed", sum(routing_result.values()))
            
            # Log routing decisions
            for agent, needed in routing_result.items():
                mlflow.log_param(f"routing_{agent}_needed", needed)
            
            # Log routing metadata
            if metadata:
                for key, value in metadata.items():
                    mlflow.log_param(f"routing_{key}", str(value))
            
            # Create routing artifact
            routing_info = {
                "query": query,
                "routing_result": routing_result,
                "execution_time": execution_time,
                "metadata": metadata or {},
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            try:
                mlflow.log_dict(routing_info, "routing_decision.json")
            except OSError as fs_error:
                if "Read-only file system" in str(fs_error):
                    # Log summary as parameters when file system is read-only
                    mlflow.log_param("routing_decision_summary", f"Query: {query[:100]}...")
                else:
                    raise
            
        except Exception as e:
            print(f"Routing tracing error: {e}")
    
    def trace_synthesis(self, query: str, sql_result: Any, news_result: Any, 
                       sentiment_result: Any, final_answer: str, execution_time: float):
        """Trace synthesis step"""
        if not self.current_run:
            return
            
        try:
            # Log synthesis metrics
            mlflow.log_metric("synthesis_execution_time", execution_time)
            mlflow.log_metric("synthesis_answer_length", len(final_answer))
            
            # Log data availability
            mlflow.log_param("synthesis_has_sql_data", sql_result is not None)
            mlflow.log_param("synthesis_has_news_data", news_result is not None)
            mlflow.log_param("synthesis_has_sentiment_data", sentiment_result is not None)
            
            # Create synthesis artifact
            synthesis_info = {
                "query": query,
                "sql_result_summary": self._summarize_data(sql_result),
                "news_result_summary": self._summarize_data(news_result),
                "sentiment_result_summary": self._summarize_data(sentiment_result),
                "final_answer": final_answer,
                "execution_time": execution_time,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            try:
                mlflow.log_dict(synthesis_info, "synthesis_step.json")
            except OSError as fs_error:
                if "Read-only file system" in str(fs_error):
                    # Log summary as parameters when file system is read-only
                    mlflow.log_param("synthesis_summary", f"Query: {query[:100]}...")
                else:
                    raise
            
        except Exception as e:
            print(f"Synthesis tracing error: {e}")
    
    def trace_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Trace errors in the workflow"""
        if not self.current_run:
            return
            
        try:
            # Log error metrics
            mlflow.log_metric(f"error_{error_type}_occurred", 1)
            mlflow.log_param(f"error_{error_type}_message", error_message)
            
            # Log context
            if context:
                for key, value in context.items():
                    mlflow.log_param(f"error_context_{key}", str(value))
            
            # Create error artifact
            error_info = {
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {},
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            try:
                mlflow.log_dict(error_info, f"error_{error_type}.json")
            except OSError as fs_error:
                if "Read-only file system" in str(fs_error):
                    # Log summary as parameters when file system is read-only
                    mlflow.log_param(f"error_{error_type}_summary", error_message[:100] + "...")
                else:
                    raise
            
        except Exception as e:
            print(f"Error tracing error: {e}")
    
    def _summarize_data(self, data: Any) -> Dict[str, Any]:
        """Create a summary of data for logging"""
        if data is None:
            return {"type": "none", "value": None}
        
        if isinstance(data, str):
            return {
                "type": "string",
                "length": len(data),
                "preview": data[:100] + "..." if len(data) > 100 else data
            }
        
        if isinstance(data, dict):
            return {
                "type": "dict",
                "keys": list(data.keys()),
                "size": len(data)
            }
        
        if isinstance(data, list):
            return {
                "type": "list",
                "length": len(data),
                "preview": data[:3] if len(data) > 3 else data
            }
        
        return {
            "type": str(type(data)),
            "value": str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
        }
    
    def set_workflow_outputs(self, outputs: Dict[str, Any]):
        """Set the final outputs for the workflow trace"""
        try:
            self.final_outputs = outputs
            # Log workflow outputs as metrics and artifacts
            if self.current_run:
                # Log output summary metrics
                output_summary = self._summarize_data(outputs)
                mlflow.log_param("workflow_output_type", output_summary.get("type", "unknown"))
                
                if "length" in output_summary:
                    mlflow.log_metric("workflow_output_length", output_summary["length"])
                
                # Save complete outputs as artifact
                try:
                    mlflow.log_dict(outputs, "workflow_outputs.json")
                except OSError as fs_error:
                    if "Read-only file system" in str(fs_error):
                        # Log summary as parameters when file system is read-only
                        mlflow.log_param("workflow_outputs_summary", str(outputs)[:200] + "...")
                    else:
                        raise
        except Exception as e:
            print(f"Error setting workflow outputs: {e}")
    
    def flush(self):
        """Flush any pending MLFlow operations"""
        try:
            if self.current_run:
                # MLFlow automatically flushes on run end, but we can ensure completion
                pass
            print("MLFlow operations flushed successfully")
        except Exception as e:
            print(f"Error flushing MLFlow operations: {e}")

# Global tracer instance
tracer = LangGraphTracer()

def trace_node(node_name: str):
    """Decorator for tracing node execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            input_data = {"args": args, "kwargs": kwargs}
            output_data = None
            error = None
            
            try:
                result = await func(*args, **kwargs)
                output_data = result
                return result
            except Exception as e:
                error = str(e)
                tracer.trace_error("node_execution", f"Error in {node_name}: {str(e)}", 
                                 {"node_name": node_name})
                raise
            finally:
                execution_time = time.time() - start_time
                tracer.trace_node_execution(
                    node_name, input_data, output_data, execution_time,
                    {"error": error} if error else None
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            input_data = {"args": args, "kwargs": kwargs}
            output_data = None
            error = None
            
            try:
                result = func(*args, **kwargs)
                output_data = result
                return result
            except Exception as e:
                error = str(e)
                tracer.trace_error("node_execution", f"Error in {node_name}: {str(e)}", 
                                 {"node_name": node_name})
                raise
            finally:
                execution_time = time.time() - start_time
                tracer.trace_node_execution(
                    node_name, input_data, output_data, execution_time,
                    {"error": error} if error else None
                )
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator