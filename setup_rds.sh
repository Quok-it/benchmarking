#!/bin/bash

# AWS RDS PostgreSQL Setup Script
# Prerequisites: AWS CLI configured with appropriate permissions

# Configuration variables
DB_INSTANCE_IDENTIFIER="gpu-benchmark-db"
DB_NAME="gpu_benchmarks"
DB_USERNAME="benchmark_user"
DB_PASSWORD="your_secure_password_here"  # Change this!
DB_INSTANCE_CLASS="db.t3.micro"  # Adjust based on your needs
DB_ENGINE="postgres"
DB_ENGINE_VERSION="15.4"
DB_ALLOCATED_STORAGE=20
DB_STORAGE_TYPE="gp2"
DB_BACKUP_RETENTION_PERIOD=7
DB_MULTI_AZ=false  # Set to true for production
DB_PUBLICLY_ACCESSIBLE=true  # Set to false for production
DB_VPC_SECURITY_GROUP_IDS="sg-xxxxxxxxx"  # Replace with your security group
DB_SUBNET_GROUP="default"  # Replace with your subnet group

# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier $DB_INSTANCE_IDENTIFIER \
    --db-instance-class $DB_INSTANCE_CLASS \
    --engine $DB_ENGINE \
    --engine-version $DB_ENGINE_VERSION \
    --master-username $DB_USERNAME \
    --master-user-password $DB_PASSWORD \
    --allocated-storage $DB_ALLOCATED_STORAGE \
    --storage-type $DB_STORAGE_TYPE \
    --backup-retention-period $DB_BACKUP_RETENTION_PERIOD \
    --multi-az $DB_MULTI_AZ \
    --publicly-accessible $DB_PUBLICLY_ACCESSIBLE \
    --vpc-security-group-ids $DB_VPC_SECURITY_GROUP_IDS \
    --db-subnet-group-name $DB_SUBNET_GROUP \
    --db-name $DB_NAME \
    --auto-minor-version-upgrade \
    --deletion-protection

echo "RDS instance creation initiated. This may take 10-15 minutes."
echo "Check status with: aws rds describe-db-instances --db-instance-identifier $DB_INSTANCE_IDENTIFIER"

# Wait for instance to be available
echo "Waiting for instance to become available..."
aws rds wait db-instance-available --db-instance-identifier $DB_INSTANCE_IDENTIFIER

# Get the endpoint
DB_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier $DB_INSTANCE_IDENTIFIER \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

echo "Database endpoint: $DB_ENDPOINT"
echo "Database name: $DB_NAME"
echo "Username: $DB_USERNAME"
echo "Password: $DB_PASSWORD"

# Create .env file with connection details
cat > .env << EOF
# Database Configuration
DB_HOST=$DB_ENDPOINT
DB_PORT=5432
DB_NAME=$DB_NAME
DB_USER=$DB_USERNAME
DB_PASSWORD=$DB_PASSWORD

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
EOF

echo "Created .env file with database configuration"
echo "Remember to add .env to your .gitignore file!" 