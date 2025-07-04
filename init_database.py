#!/usr/bin/env python3
"""
Database initialization script for GPU Benchmarking System
"""

import os
import psycopg2
from dotenv import load_dotenv

def init_database():
    """Initialize the database with the schema"""
    load_dotenv()
    
    # Connect to PostgreSQL
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        print("Successfully connected to PostgreSQL database")
        
        # Read and execute schema
        with open('database_schema.sql', 'r') as f:
            schema_sql = f.read()
        
        with connection.cursor() as cursor:
            cursor.execute(schema_sql)
        
        connection.commit()
        print("Database schema initialized successfully")
        
        # Test the connection and basic queries
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM raw_benchmark_results")
            count = cursor.fetchone()[0]
            print(f"Raw benchmark results table: {count} records")
            
            cursor.execute("SELECT COUNT(*) FROM gpu_aggregates")
            count = cursor.fetchone()[0]
            print(f"GPU aggregates table: {count} records")
            
            cursor.execute("SELECT COUNT(*) FROM audit_progress")
            count = cursor.fetchone()[0]
            print(f"Audit progress table: {count} records")
        
        connection.close()
        print("Database initialization completed successfully!")
        
    except Exception as e:
        print(f"Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    init_database() 