#!/usr/bin/env python3
"""
test script to check connection to AWS RDS PostgreSQL using .env configuration
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
import socket

def test_env_file():
    """Test if .env file exists and has required variables"""
    print("=== Testing .env File ===")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        return False
    
    load_dotenv()
    
    required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            # mask password
            if var == 'DB_PASSWORD':
                print(f"✅ {var}: {'*' * len(value)}")
            else:
                print(f"✅ {var}: {value}")
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    print("✅ All required environment variables found")
    return True

def test_network_connectivity():
    """Test network connectivity to db host"""
    print("\n=== Testing Network Connectivity ===")
    
    host = os.getenv('DB_HOST')
    port = int(os.getenv('DB_PORT', '5432'))
    
    try:
        # test DNS resolution
        print(f"Testing DNS resolution for {host}...")
        ip_address = socket.gethostbyname(host)
        print(f"✅ DNS resolution successful: {host} -> {ip_address}")
        
        # test TCP connection
        print(f"Testing TCP connection to {host}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ TCP connection successful to {host}:{port}")
            return True
        else:
            print(f"❌ TCP connection failed to {host}:{port}")
            print("This might indicate:")
            print("  - Security group not configured properly")
            print("  - Database not publicly accessible")
            print("  - Firewall blocking connection")
            return False
            
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        return False

def test_database_connection():
    """Test actual db connection"""
    print("\n=== Testing Database Connection ===")
    
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            connect_timeout=10
        )
        
        print("✅ Database connection established successfully!")
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Database connection failed: {e}")
        return False
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Run all connection tests"""
    print("Testing AWS RDS PostgreSQL Connection")
    print("=" * 50)
    
    # test 1: env file
    if not test_env_file():
        sys.exit(1)
    
    # test 2: network connectivity
    if not test_network_connectivity():
        print("\n⚠️  Network connectivity failed. Please check:")
        print("  - AWS RDS instance status")
        print("  - Security group configuration")
        print("  - VPC and subnet settings")
        sys.exit(1)
    
    # test 3: db connection
    if not test_database_connection():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("All tests passed! Database connection is working correctly.")

if __name__ == "__main__":
    main() 