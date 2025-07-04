# PostgreSQL Migration Setup Guide

This guide explains how to migrate from MongoDB to AWS RDS PostgreSQL for the GPU benchmarking system.

## Architecture Overview

The new system implements an **event-driven incremental computation** architecture:

```
Raw Data Storage → Event Stream → Analytics Engine → Live Results
     ↓                ↓              ↓               ↓
PostgreSQL    →  Change Events  →  Aggregation   →  Dashboard
                                    Logic          Real-time Queries
```

## 1. AWS RDS PostgreSQL Setup

### Prerequisites
- AWS CLI configured with appropriate permissions
- VPC and security groups configured
- Subnet group created

### Step 1: Create RDS Instance

```bash
# Make the setup script executable
chmod +x setup_rds.sh

# Edit the script to configure your settings
nano setup_rds.sh

# Run the setup script
./setup_rds.sh
```

**Important Configuration Notes:**
- Update `DB_PASSWORD` with a secure password
- Set `DB_VPC_SECURITY_GROUP_IDS` to your security group ID
- Set `DB_SUBNET_GROUP` to your subnet group name
- For production, set `DB_PUBLICLY_ACCESSIBLE=false`
- For production, set `DB_MULTI_AZ=true`

### Step 2: Configure Security Groups

Ensure your RDS security group allows connections from your benchmark servers:
- Port 5432 (PostgreSQL)
- Source: Your benchmark server IP or security group

## 2. Database Schema

The schema implements your architectural requirements:

### Core Tables

1. **`raw_benchmark_results`** (Append-Only)
   - Stores all raw benchmark data
   - Uses JSONB for flexible data storage
   - Includes processing status and retry logic

2. **`gpu_aggregates`** (Updated Incrementally)
   - Maintains running statistics for each GPU/metric combination
   - Supports incremental updates without full recalculation
   - Includes reliability scores

3. **`audit_progress`** (Updated Incrementally)
   - Tracks audit completion status
   - Provides real-time progress monitoring

4. **`event_log`** (Event Stream)
   - Captures benchmark completion events
   - Enables event-driven processing

### Views for Analytics

- **`gpu_performance_summary`**: Real-time performance metrics
- **`audit_progress_summary`**: Current audit status

### Functions for Incremental Processing

- **`update_gpu_aggregates()`**: Incrementally updates statistics
- **`emit_event()`**: Creates events for processing

## 3. Code Changes

### Dependencies Updated

**Old (MongoDB):**
```bash
pip install pymongo
```

**New (PostgreSQL):**
```bash
pip install psycopg2-binary python-dotenv
```

### Database Connection

**Old (MongoDB):**
```python
from pymongo import MongoClient
client = MongoClient(os.environ["MONGODB_URI"])
db = client["QCP"]
```

**New (PostgreSQL):**
```python
import psycopg2
connection = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
```

### Data Storage Pattern

**Old (MongoDB):**
```python
result_doc = {
    "mlperf_results": mlperf_results,
    "gpu_burn_result": gpu_status,
    "hpc_results": {...}
}
db.benchmark_results.insert_one(result_doc)
```

**New (PostgreSQL):**
```python
# Raw data storage
db.insert_benchmark_result(
    hostname, gpu_id, gpu_model, 'mlperf', benchmark_data
)

# Incremental aggregates
db.update_gpu_aggregates(
    hostname, gpu_id, gpu_model, 'mlperf', metrics
)
```

## 4. Setup Process

### Step 1: Environment Configuration

The `setup_rds.sh` script creates a `.env` file:

```env
# Database Configuration
DB_HOST=your-rds-endpoint.amazonaws.com
DB_PORT=5432
DB_NAME=gpu_benchmarks
DB_USER=benchmark_user
DB_PASSWORD=your_secure_password

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Step 2: Database Initialization

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database schema
python3 init_database.py
```

### Step 3: Run Benchmarks

```bash
# Run the complete benchmark suite
./benchmarks.sh
```

## 5. Event-Driven Processing

### Automatic Event Generation

The system automatically generates events when benchmark results are inserted:

```sql
-- Trigger creates events automatically
CREATE TRIGGER benchmark_event_trigger
    AFTER INSERT ON raw_benchmark_results
    FOR EACH ROW
    EXECUTE FUNCTION trigger_benchmark_event();
```

### Event Processing

Events can be processed for:
- Real-time dashboard updates
- Alert generation
- Incremental analytics updates
- Audit progress tracking

## 6. Querying Data

### Real-Time Analytics

```sql
-- Get current GPU performance summary
SELECT * FROM gpu_performance_summary 
WHERE gpu_node = 'your-hostname';

-- Get audit progress
SELECT * FROM audit_progress_summary 
WHERE status = 'running';

-- Get recent benchmark results
SELECT * FROM raw_benchmark_results 
WHERE timestamp > NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;
```

### Incremental Aggregates

```sql
-- Get reliability scores
SELECT gpu_node, gpu_id, 
       AVG(reliability_score) as avg_reliability
FROM gpu_aggregates 
WHERE benchmark_type = 'gpu_burn'
GROUP BY gpu_node, gpu_id;
```

## 7. Monitoring and Maintenance

### Database Monitoring

- Monitor RDS CloudWatch metrics
- Set up alerts for connection issues
- Track query performance

### Data Retention

Consider implementing data retention policies:
```sql
-- Archive old raw data (example)
DELETE FROM raw_benchmark_results 
WHERE timestamp < NOW() - INTERVAL '90 days';
```

## 8. Migration Checklist

- [ ] AWS RDS instance created and accessible
- [ ] Security groups configured
- [ ] Database schema initialized
- [ ] Environment variables configured
- [ ] Dependencies updated
- [ ] Code migrated to PostgreSQL
- [ ] Database connection tested
- [ ] Benchmark results verified
- [ ] Event processing tested
- [ ] Analytics queries validated

## 9. Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check security group settings
   - Verify RDS endpoint
   - Ensure database is publicly accessible (if needed)

2. **Authentication Failed**
   - Verify username/password in `.env`
   - Check database name

3. **Schema Errors**
   - Run `init_database.py` to recreate schema
   - Check PostgreSQL version compatibility

### Debug Commands

```bash
# Test database connection
python3 -c "
import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
print('Connection successful!')
conn.close()
"
```

## 10. Next Steps

After successful migration:

1. **Set up monitoring dashboards** (Grafana)
2. **Implement event processors** for real-time analytics
3. **Create automated reports** (PDF generation)
4. **Set up backup and recovery** procedures
5. **Implement data archival** strategies 