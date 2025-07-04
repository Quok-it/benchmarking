-- GPU Benchmark Database Schema
-- Implements event-driven incremental computation architecture

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================
-- RAW DATA STORAGE (Append-Only)
-- =============================================

-- Raw benchmark results table
CREATE TABLE raw_benchmark_results (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    gpu_node VARCHAR(100) NOT NULL,
    gpu_id VARCHAR(50) NOT NULL,
    gpu_model VARCHAR(100),
    benchmark_type VARCHAR(50) NOT NULL, -- 'mlperf', 'gpu_burn', 'hpl', 'hpcg', 'stream'
    benchmark_data JSONB NOT NULL, -- Flexible storage for different benchmark types
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processing_status VARCHAR(20) DEFAULT 'new' CHECK (processing_status IN ('new', 'processed', 'failed')),
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX idx_raw_benchmark_gpu ON raw_benchmark_results(gpu_node, gpu_id);
CREATE INDEX idx_raw_benchmark_type ON raw_benchmark_results(benchmark_type);
CREATE INDEX idx_raw_benchmark_status ON raw_benchmark_results(processing_status);
CREATE INDEX idx_raw_benchmark_timestamp ON raw_benchmark_results(timestamp);

-- =============================================
-- GPU AGGREGATES (Updated Incrementally)
-- =============================================

-- GPU performance aggregates table
CREATE TABLE gpu_aggregates (
    gpu_node VARCHAR(100) NOT NULL,
    gpu_id VARCHAR(50) NOT NULL,
    gpu_model VARCHAR(100),
    benchmark_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL, -- 'samples_per_sec', 'latency', 'gflops', etc.
    count_measurements INTEGER DEFAULT 0,
    running_sum DOUBLE PRECISION DEFAULT 0,
    running_sum_squares DOUBLE PRECISION DEFAULT 0,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    last_value DOUBLE PRECISION,
    reliability_score DOUBLE PRECISION DEFAULT 1.0, -- 0-1 scale
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (gpu_node, gpu_id, benchmark_type, metric_name)
);

-- Indexes for aggregates
CREATE INDEX idx_gpu_aggregates_gpu ON gpu_aggregates(gpu_node, gpu_id);
CREATE INDEX idx_gpu_aggregates_type ON gpu_aggregates(benchmark_type);
CREATE INDEX idx_gpu_aggregates_reliability ON gpu_aggregates(reliability_score);

-- =============================================
-- AUDIT PROGRESS (Updated Incrementally)
-- =============================================

-- Audit progress tracking table
CREATE TABLE audit_progress (
    audit_session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audit_name VARCHAR(100) NOT NULL,
    total_gpus INTEGER DEFAULT 0,
    completed_gpus INTEGER DEFAULT 0,
    failed_gpus INTEGER DEFAULT 0,
    completion_percentage DECIMAL(5,2) DEFAULT 0.0,
    start_time TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'paused')),
    metadata JSONB -- Additional audit metadata
);

-- =============================================
-- EVENT LOG (For Event-Driven Processing)
-- =============================================

-- Event log table for tracking benchmark completions
CREATE TABLE event_log (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL, -- 'gpu_benchmark_completed', 'audit_started', etc.
    audit_id UUID REFERENCES raw_benchmark_results(audit_id),
    gpu_node VARCHAR(100),
    gpu_id VARCHAR(50),
    event_data JSONB,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

-- Indexes for event processing
CREATE INDEX idx_event_log_type ON event_log(event_type);
CREATE INDEX idx_event_log_processed ON event_log(processed);
CREATE INDEX idx_event_log_created ON event_log(created_at);

-- =============================================
-- VIEWS FOR ANALYTICS
-- =============================================

-- View for current GPU performance summary
CREATE VIEW gpu_performance_summary AS
SELECT 
    gpu_node,
    gpu_id,
    gpu_model,
    benchmark_type,
    metric_name,
    count_measurements,
    CASE 
        WHEN count_measurements > 0 THEN running_sum / count_measurements 
        ELSE NULL 
    END as avg_value,
    CASE 
        WHEN count_measurements > 1 THEN 
            SQRT((running_sum_squares - (running_sum * running_sum / count_measurements)) / (count_measurements - 1))
        ELSE NULL 
    END as std_deviation,
    min_value,
    max_value,
    last_value,
    reliability_score,
    last_updated
FROM gpu_aggregates;

-- View for audit progress summary
CREATE VIEW audit_progress_summary AS
SELECT 
    audit_session_id,
    audit_name,
    total_gpus,
    completed_gpus,
    failed_gpus,
    completion_percentage,
    start_time,
    last_updated,
    status,
    CASE 
        WHEN total_gpus > 0 THEN 
            ROUND((completed_gpus::DECIMAL / total_gpus) * 100, 2)
        ELSE 0 
    END as calculated_percentage
FROM audit_progress;

-- =============================================
-- FUNCTIONS FOR INCREMENTAL PROCESSING
-- =============================================

-- Function to update GPU aggregates incrementally
CREATE OR REPLACE FUNCTION update_gpu_aggregates(
    p_gpu_node VARCHAR,
    p_gpu_id VARCHAR,
    p_gpu_model VARCHAR,
    p_benchmark_type VARCHAR,
    p_metric_name VARCHAR,
    p_value DOUBLE PRECISION
) RETURNS VOID AS $$
BEGIN
    INSERT INTO gpu_aggregates (
        gpu_node, gpu_id, gpu_model, benchmark_type, metric_name,
        count_measurements, running_sum, running_sum_squares,
        min_value, max_value, last_value, last_updated
    ) VALUES (
        p_gpu_node, p_gpu_id, p_gpu_model, p_benchmark_type, p_metric_name,
        1, p_value, p_value * p_value,
        p_value, p_value, p_value, NOW()
    )
    ON CONFLICT (gpu_node, gpu_id, benchmark_type, metric_name)
    DO UPDATE SET
        count_measurements = gpu_aggregates.count_measurements + 1,
        running_sum = gpu_aggregates.running_sum + p_value,
        running_sum_squares = gpu_aggregates.running_sum_squares + (p_value * p_value),
        min_value = LEAST(gpu_aggregates.min_value, p_value),
        max_value = GREATEST(gpu_aggregates.max_value, p_value),
        last_value = p_value,
        last_updated = NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to emit events
CREATE OR REPLACE FUNCTION emit_event(
    p_event_type VARCHAR,
    p_audit_id UUID,
    p_gpu_node VARCHAR,
    p_gpu_id VARCHAR,
    p_event_data JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_event_id UUID;
BEGIN
    INSERT INTO event_log (event_type, audit_id, gpu_node, gpu_id, event_data)
    VALUES (p_event_type, p_audit_id, p_gpu_node, p_gpu_id, p_event_data)
    RETURNING event_id INTO v_event_id;
    
    RETURN v_event_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- TRIGGERS FOR AUTOMATIC EVENT GENERATION
-- =============================================

-- Trigger to automatically emit events when new benchmark results are inserted
CREATE OR REPLACE FUNCTION trigger_benchmark_event() RETURNS TRIGGER AS $$
BEGIN
    PERFORM emit_event(
        'gpu_benchmark_completed',
        NEW.audit_id,
        NEW.gpu_node,
        NEW.gpu_id,
        jsonb_build_object(
            'benchmark_type', NEW.benchmark_type,
            'timestamp', NEW.timestamp
        )
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER benchmark_event_trigger
    AFTER INSERT ON raw_benchmark_results
    FOR EACH ROW
    EXECUTE FUNCTION trigger_benchmark_event();

-- =============================================
-- SAMPLE DATA INSERTION
-- =============================================

-- Insert a sample audit session
INSERT INTO audit_progress (audit_name, total_gpus, status) 
VALUES ('GPU Reliability Audit 2024', 0, 'running');

-- =============================================
-- GRANTS (if using separate users)
-- =============================================

-- Grant permissions to benchmark user
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO benchmark_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO benchmark_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO benchmark_user; 