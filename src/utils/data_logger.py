"""
Data Logger Module for Sustainable AI

This module provides logging and reporting functionality for:
- Prompt analysis sessions
- Energy estimates
- Optimization results
- User interactions

Supports both SQLite (lightweight) and PostgreSQL (scalable) backends.

Author: Sustainable AI Team
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)


@dataclass
class AnalysisLog:
    """Data class for analysis session logs."""
    session_id: str
    timestamp: str
    prompt_text: str
    prompt_hash: str  # For privacy - can log hash instead of full prompt
    original_tokens: int
    optimized_tokens: int
    token_reduction_pct: float
    energy_kwh: float
    carbon_kg: float
    optimized_energy_kwh: float
    optimized_carbon_kg: float
    energy_saved_pct: float
    semantic_similarity: float
    quality_score: float
    model_type: str
    layers: int
    training_hours: float
    flops: str
    recommendation_chosen: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class DataLogger:
    """
    Data Logger for Sustainable AI analysis sessions.
    
    Supports:
    - SQLite (default, lightweight)
    - PostgreSQL (scalable, requires psycopg2)
    - JSON file logging (fallback)
    """
    
    def __init__(
        self, 
        db_type: str = "sqlite",
        db_path: Optional[str] = None,
        pg_config: Optional[Dict] = None
    ):
        """
        Initialize the DataLogger.
        
        Args:
            db_type: Database type ('sqlite', 'postgresql', 'json')
            db_path: Path to SQLite database or JSON log file
            pg_config: PostgreSQL configuration dictionary
        """
        self.db_type = db_type
        self.pg_config = pg_config
        
        # Set default paths
        if db_path is None:
            data_dir = os.path.join(project_root, 'data', 'logs')
            os.makedirs(data_dir, exist_ok=True)
            if db_type == "sqlite":
                db_path = os.path.join(data_dir, 'analysis_logs.db')
            else:
                db_path = os.path.join(data_dir, 'analysis_logs.json')
        
        self.db_path = db_path
        
        # Initialize database
        if db_type == "sqlite":
            self._init_sqlite()
        elif db_type == "postgresql":
            self._init_postgresql()
        else:
            self._init_json()
    
    def _init_sqlite(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main analysis log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                timestamp TEXT,
                prompt_text TEXT,
                prompt_hash TEXT,
                original_tokens INTEGER,
                optimized_tokens INTEGER,
                token_reduction_pct REAL,
                energy_kwh REAL,
                carbon_kg REAL,
                optimized_energy_kwh REAL,
                optimized_carbon_kg REAL,
                energy_saved_pct REAL,
                semantic_similarity REAL,
                quality_score REAL,
                model_type TEXT,
                layers INTEGER,
                training_hours REAL,
                flops TEXT,
                recommendation_chosen TEXT
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_logs(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_prompt_hash ON analysis_logs(prompt_hash)
        ''')
        
        # Create summary statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                total_analyses INTEGER DEFAULT 0,
                total_energy_saved_kwh REAL DEFAULT 0,
                total_carbon_saved_kg REAL DEFAULT 0,
                avg_token_reduction REAL DEFAULT 0,
                avg_quality_score REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"SQLite database initialized: {self.db_path}")
    
    def _init_postgresql(self):
        """Initialize PostgreSQL database."""
        try:
            import psycopg2
            
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_logs (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(64) UNIQUE,
                    timestamp TIMESTAMP,
                    prompt_text TEXT,
                    prompt_hash VARCHAR(64),
                    original_tokens INTEGER,
                    optimized_tokens INTEGER,
                    token_reduction_pct FLOAT,
                    energy_kwh FLOAT,
                    carbon_kg FLOAT,
                    optimized_energy_kwh FLOAT,
                    optimized_carbon_kg FLOAT,
                    energy_saved_pct FLOAT,
                    semantic_similarity FLOAT,
                    quality_score FLOAT,
                    model_type VARCHAR(50),
                    layers INTEGER,
                    training_hours FLOAT,
                    flops VARCHAR(20),
                    recommendation_chosen VARCHAR(100)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("PostgreSQL database initialized")
        except ImportError:
            print("psycopg2 not installed. Falling back to SQLite.")
            self.db_type = "sqlite"
            self._init_sqlite()
    
    def _init_json(self):
        """Initialize JSON file logging."""
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                json.dump({"logs": [], "stats": {}}, f)
        print(f"JSON log file initialized: {self.db_path}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().isoformat()
        unique_str = f"{timestamp}-{os.urandom(8).hex()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]
    
    def _hash_prompt(self, prompt: str) -> str:
        """Generate hash of prompt for privacy."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]
    
    def log_analysis(
        self,
        prompt_text: str,
        original_tokens: int,
        optimized_tokens: int,
        token_reduction_pct: float,
        energy_kwh: float,
        carbon_kg: float,
        optimized_energy_kwh: float,
        optimized_carbon_kg: float,
        semantic_similarity: float,
        quality_score: float,
        model_type: str,
        layers: int,
        training_hours: float,
        flops: str,
        recommendation_chosen: Optional[str] = None,
        store_full_prompt: bool = False
    ) -> str:
        """
        Log an analysis session.
        
        Args:
            prompt_text: The original prompt (or anonymized version)
            original_tokens: Original token count
            optimized_tokens: Optimized token count
            token_reduction_pct: Percentage reduction
            energy_kwh: Original energy estimate
            carbon_kg: Original carbon footprint
            optimized_energy_kwh: Optimized energy estimate
            optimized_carbon_kg: Optimized carbon footprint
            semantic_similarity: Similarity score
            quality_score: Overall quality score
            model_type: ML model used
            layers: Number of layers
            training_hours: Training time
            flops: FLOPs value
            recommendation_chosen: Which recommendation user chose
            store_full_prompt: Whether to store full prompt (privacy)
            
        Returns:
            Session ID
        """
        session_id = self._generate_session_id()
        timestamp = datetime.now().isoformat()
        prompt_hash = self._hash_prompt(prompt_text)
        
        # Calculate energy saved percentage
        energy_saved_pct = 0.0
        if energy_kwh > 0:
            energy_saved_pct = (energy_kwh - optimized_energy_kwh) / energy_kwh * 100
        
        log_entry = AnalysisLog(
            session_id=session_id,
            timestamp=timestamp,
            prompt_text=prompt_text if store_full_prompt else f"[HASH:{prompt_hash[:8]}]",
            prompt_hash=prompt_hash,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            token_reduction_pct=token_reduction_pct,
            energy_kwh=energy_kwh,
            carbon_kg=carbon_kg,
            optimized_energy_kwh=optimized_energy_kwh,
            optimized_carbon_kg=optimized_carbon_kg,
            energy_saved_pct=round(energy_saved_pct, 2),
            semantic_similarity=semantic_similarity,
            quality_score=quality_score,
            model_type=model_type,
            layers=layers,
            training_hours=training_hours,
            flops=flops,
            recommendation_chosen=recommendation_chosen
        )
        
        if self.db_type == "sqlite":
            self._log_to_sqlite(log_entry)
        elif self.db_type == "postgresql":
            self._log_to_postgresql(log_entry)
        else:
            self._log_to_json(log_entry)
        
        # Update daily statistics
        self._update_daily_stats(log_entry)
        
        return session_id
    
    def _log_to_sqlite(self, log: AnalysisLog):
        """Log entry to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_logs (
                session_id, timestamp, prompt_text, prompt_hash,
                original_tokens, optimized_tokens, token_reduction_pct,
                energy_kwh, carbon_kg, optimized_energy_kwh, optimized_carbon_kg,
                energy_saved_pct, semantic_similarity, quality_score,
                model_type, layers, training_hours, flops, recommendation_chosen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log.session_id, log.timestamp, log.prompt_text, log.prompt_hash,
            log.original_tokens, log.optimized_tokens, log.token_reduction_pct,
            log.energy_kwh, log.carbon_kg, log.optimized_energy_kwh, log.optimized_carbon_kg,
            log.energy_saved_pct, log.semantic_similarity, log.quality_score,
            log.model_type, log.layers, log.training_hours, log.flops, log.recommendation_chosen
        ))
        
        conn.commit()
        conn.close()
    
    def _log_to_postgresql(self, log: AnalysisLog):
        """Log entry to PostgreSQL."""
        import psycopg2
        
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_logs (
                session_id, timestamp, prompt_text, prompt_hash,
                original_tokens, optimized_tokens, token_reduction_pct,
                energy_kwh, carbon_kg, optimized_energy_kwh, optimized_carbon_kg,
                energy_saved_pct, semantic_similarity, quality_score,
                model_type, layers, training_hours, flops, recommendation_chosen
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            log.session_id, log.timestamp, log.prompt_text, log.prompt_hash,
            log.original_tokens, log.optimized_tokens, log.token_reduction_pct,
            log.energy_kwh, log.carbon_kg, log.optimized_energy_kwh, log.optimized_carbon_kg,
            log.energy_saved_pct, log.semantic_similarity, log.quality_score,
            log.model_type, log.layers, log.training_hours, log.flops, log.recommendation_chosen
        ))
        
        conn.commit()
        conn.close()
    
    def _log_to_json(self, log: AnalysisLog):
        """Log entry to JSON file."""
        with open(self.db_path, 'r') as f:
            data = json.load(f)
        
        data['logs'].append(log.to_dict())
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_daily_stats(self, log: AnalysisLog):
        """Update daily statistics."""
        today = datetime.now().strftime('%Y-%m-%d')
        energy_saved = log.energy_kwh - log.optimized_energy_kwh
        carbon_saved = log.carbon_kg - log.optimized_carbon_kg
        
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO daily_stats (date, total_analyses, total_energy_saved_kwh, 
                    total_carbon_saved_kg, avg_token_reduction, avg_quality_score)
                VALUES (?, 1, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_analyses = total_analyses + 1,
                    total_energy_saved_kwh = total_energy_saved_kwh + ?,
                    total_carbon_saved_kg = total_carbon_saved_kg + ?,
                    avg_token_reduction = (avg_token_reduction * (total_analyses - 1) + ?) / total_analyses,
                    avg_quality_score = (avg_quality_score * (total_analyses - 1) + ?) / total_analyses
            ''', (
                today, energy_saved, carbon_saved, log.token_reduction_pct, log.quality_score,
                energy_saved, carbon_saved, log.token_reduction_pct, log.quality_score
            ))
            
            conn.commit()
            conn.close()
    
    def get_session_log(self, session_id: str) -> Optional[Dict]:
        """Retrieve a specific session log."""
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM analysis_logs WHERE session_id = ?', (session_id,))
            row = cursor.fetchone()
            conn.close()
            
            return dict(row) if row else None
        return None
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent analysis logs."""
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM analysis_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        return []
    
    def get_statistics(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """
        Get aggregate statistics.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with statistics
        """
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT 
                    COUNT(*) as total_analyses,
                    AVG(token_reduction_pct) as avg_token_reduction,
                    AVG(energy_saved_pct) as avg_energy_saved,
                    SUM(energy_kwh - optimized_energy_kwh) as total_energy_saved,
                    SUM(carbon_kg - optimized_carbon_kg) as total_carbon_saved,
                    AVG(quality_score) as avg_quality_score,
                    AVG(semantic_similarity) as avg_semantic_similarity
                FROM analysis_logs
            '''
            
            params = []
            if start_date or end_date:
                conditions = []
                if start_date:
                    conditions.append("timestamp >= ?")
                    params.append(start_date)
                if end_date:
                    conditions.append("timestamp <= ?")
                    params.append(end_date + "T23:59:59")
                query += " WHERE " + " AND ".join(conditions)
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "total_analyses": row[0] or 0,
                    "avg_token_reduction_pct": round(row[1] or 0, 2),
                    "avg_energy_saved_pct": round(row[2] or 0, 2),
                    "total_energy_saved_kwh": round(row[3] or 0, 4),
                    "total_carbon_saved_kg": round(row[4] or 0, 4),
                    "avg_quality_score": round(row[5] or 0, 2),
                    "avg_semantic_similarity": round(row[6] or 0, 2)
                }
        
        return {}
    
    def generate_report(self, format: str = "text") -> str:
        """
        Generate a summary report.
        
        Args:
            format: Output format ('text', 'json', 'markdown')
            
        Returns:
            Formatted report string
        """
        stats = self.get_statistics()
        
        if format == "json":
            return json.dumps(stats, indent=2)
        
        elif format == "markdown":
            return f"""
# Sustainable AI Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Analyses | {stats.get('total_analyses', 0)} |
| Avg Token Reduction | {stats.get('avg_token_reduction_pct', 0)}% |
| Avg Energy Saved | {stats.get('avg_energy_saved_pct', 0)}% |
| Total Energy Saved | {stats.get('total_energy_saved_kwh', 0)} kWh |
| Total Carbon Saved | {stats.get('total_carbon_saved_kg', 0)} kg CO₂ |
| Avg Quality Score | {stats.get('avg_quality_score', 0)} |
| Avg Semantic Similarity | {stats.get('avg_semantic_similarity', 0)}% |

## Environmental Impact

The optimizations have helped save an estimated **{stats.get('total_energy_saved_kwh', 0):.4f} kWh** of energy,
equivalent to **{stats.get('total_carbon_saved_kg', 0) * 1000:.2f} grams of CO₂**.
"""
        
        else:  # text
            return f"""
============================================================
         SUSTAINABLE AI ANALYSIS REPORT
============================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
------------------
Total Analyses:        {stats.get('total_analyses', 0)}
Avg Token Reduction:   {stats.get('avg_token_reduction_pct', 0)}%
Avg Energy Saved:      {stats.get('avg_energy_saved_pct', 0)}%
Total Energy Saved:    {stats.get('total_energy_saved_kwh', 0):.4f} kWh
Total Carbon Saved:    {stats.get('total_carbon_saved_kg', 0):.4f} kg CO₂
Avg Quality Score:     {stats.get('avg_quality_score', 0)}
Avg Semantic Sim:      {stats.get('avg_semantic_similarity', 0)}%

============================================================
"""
    
    def export_logs(self, filepath: str, format: str = "csv") -> bool:
        """
        Export logs to file.
        
        Args:
            filepath: Output file path
            format: Export format ('csv', 'json')
            
        Returns:
            True if successful
        """
        logs = self.get_recent_logs(limit=10000)
        
        if format == "csv":
            import csv
            
            if logs:
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                    writer.writeheader()
                    writer.writerows(logs)
                return True
        
        elif format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2)
            return True
        
        return False


# Convenience function for quick logging
def log_analysis(**kwargs) -> str:
    """Quick function to log an analysis session."""
    logger = DataLogger()
    return logger.log_analysis(**kwargs)


if __name__ == "__main__":
    # Test the logger
    logger = DataLogger(db_type="sqlite")
    
    # Log a test entry
    session_id = logger.log_analysis(
        prompt_text="Test prompt for logging",
        original_tokens=10,
        optimized_tokens=7,
        token_reduction_pct=30.0,
        energy_kwh=1.5,
        carbon_kg=0.7,
        optimized_energy_kwh=1.2,
        optimized_carbon_kg=0.56,
        semantic_similarity=85.0,
        quality_score=75.0,
        model_type="LinearRegression",
        layers=12,
        training_hours=5.0,
        flops="1.5e18"
    )
    
    print(f"Logged session: {session_id}")
    print("\nStatistics:")
    print(logger.generate_report())
