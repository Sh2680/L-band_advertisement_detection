import mysql.connector
from mysql.connector import Error, pooling
from datetime import datetime
import logging
from config import DATABASE_CONFIG

logger = logging.getLogger(__name__)


class DatabaseManager:    
    def __init__(self):
        self.pool = None
        self.initialize_pool()
        self.create_schema()
    
    def initialize_pool(self):
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name=DATABASE_CONFIG['pool_name'],
                pool_size=DATABASE_CONFIG['pool_size'],
                host=DATABASE_CONFIG['host'],
                user=DATABASE_CONFIG['user'],
                password=DATABASE_CONFIG['password'],
                database=DATABASE_CONFIG['database'],
                autocommit=DATABASE_CONFIG['autocommit'],
                connection_timeout=DATABASE_CONFIG['connection_timeout']
            )
            logger.info("[OK] Database connection pool initialized")
        except Error as e:
            logger.error(f"[ERROR] Failed to create connection pool: {e}")
            raise
    
    def get_connection(self):
        try:
            return self.pool.get_connection()
        except Error as e:
            logger.error(f"[ERROR] Failed to get connection from pool: {e}")
            return None
    
    def create_schema(self):
        conn = self.get_connection()
        if not conn:
            logger.error("[ERROR] Cannot create schema: No database connection")
            return False
        
        try:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS lband_ad_events (
                id INT AUTO_INCREMENT PRIMARY KEY,
                track_id INT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME NOT NULL,
                duration_seconds FLOAT NOT NULL,
                appearance_count INT NOT NULL DEFAULT 1,
                INDEX idx_track_id (track_id),
                INDEX idx_start_time (start_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            cursor.execute(create_table_query)
            logger.info("[OK] Database schema validated/created")
            
            cursor.close()
            conn.close()
            return True
            
        except Error as e:
            logger.error(f"[ERROR] Failed to create schema: {e}")
            if conn:
                conn.close()
            return False
    
    def get_track_appearance_count(self, track_id):
        """Get the number of times this specific track_id has been logged"""
        conn = self.get_connection()
        if not conn:
            return 0
        try:
            cursor = conn.cursor()
            query = "SELECT COUNT(*) FROM lband_ad_events WHERE track_id = %s"
            cursor.execute(query, (int(track_id),))
            result = cursor.fetchone()
            count = result[0] if result else 0
            cursor.close()
            conn.close()
            return count
        except Error as e:
            logger.error(f"[ERROR] Failed to fetch appearance count for track {track_id}: {e}")
            if conn:
                conn.close()
            return 0

    def insert_ad_event(self, track_id, start_time, end_time, duration_seconds):
        """
        Record an L-band ad event:
        - If this track_id was already detected TODAY:
          Update the existing row in-place: increment appearance_count (+1), add duration, update end_time.
        - If this track_id is new today (or detected tomorrow on a new day):
          Insert a new row starting with appearance_count = 1.
        """
        conn = self.get_connection()
        if not conn:
            logger.error("[ERROR] Cannot save ad event: No database connection")
            return None
        
        try:
            # Disable autocommit so we can control the transaction
            conn.autocommit = False
            cursor = conn.cursor(dictionary=True)
            
            # Lock the row for this track_id today so no other thread can read stale data
            check_query = """
            SELECT id, duration_seconds, appearance_count 
            FROM lband_ad_events 
            WHERE track_id = %s AND DATE(start_time) = CURDATE()
            ORDER BY id ASC
            LIMIT 1
            FOR UPDATE
            """
            cursor.execute(check_query, (int(track_id),))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing row for today in-place: increment count, add duration, update end_time
                row_id = existing['id']
                new_count = existing['appearance_count'] + 1
                new_duration = existing['duration_seconds'] + float(duration_seconds)
                
                update_query = """
                UPDATE lband_ad_events 
                SET end_time = %s,
                    duration_seconds = %s,
                    appearance_count = %s
                WHERE id = %s
                """
                cursor.execute(update_query, (end_time, new_duration, new_count, row_id))
                conn.commit()
                
                logger.info(
                    f"[OK] Updated L-Band Ad | Row ID: {row_id} | Track ID: {track_id} | "
                    f"Appearance Count: {new_count} | Total Duration Today: {new_duration:.2f}s"
                )
                result_id = row_id
            else:
                # First time this track_id appears today -> Insert new row with count = 1
                insert_query = """
                INSERT INTO lband_ad_events 
                (track_id, start_time, end_time, duration_seconds, appearance_count)
                VALUES (%s, %s, %s, %s, 1)
                """
                cursor.execute(insert_query, (int(track_id), start_time, end_time, float(duration_seconds)))
                conn.commit()
                result_id = cursor.lastrowid
                
                logger.info(
                    f"[OK] Logged New L-Band Ad | ID: {track_id} | "
                    f"Duration: {duration_seconds:.2f}s | "
                    f"Appearance Count: 1"
                )
            
            cursor.close()
            conn.close()
            return result_id
            
        except Error as e:
            logger.error(f"[ERROR] Failed to save ad event: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
            if conn:
                conn.close()
            return None
    
    def get_recent_events(self, limit=10):
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT * FROM lband_ad_events 
            ORDER BY created_at DESC 
            LIMIT %s
            """
            
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            return results
            
        except Error as e:
            logger.error(f"[ERROR] Failed to fetch recent events: {e}")
            if conn:
                conn.close()
            return []
    
    def get_daily_occurrences(self, target_date=None):
        """Get the number of times advertisements occurred today (or on target_date)"""
        conn = self.get_connection()
        if not conn:
            return 0
        try:
            cursor = conn.cursor()
            if target_date is None:
                query = "SELECT COUNT(*) FROM lband_ad_events WHERE DATE(start_time) = CURDATE()"
                cursor.execute(query)
            else:
                query = "SELECT COUNT(*) FROM lband_ad_events WHERE DATE(start_time) = %s"
                cursor.execute(query, (target_date,))
            result = cursor.fetchone()
            count = result[0] if result else 0
            cursor.close()
            conn.close()
            return count
        except Error as e:
            logger.error(f"[ERROR] Failed to fetch daily occurrences: {e}")
            if conn:
                conn.close()
            return 0

    def get_statistics(self):
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT track_id) as unique_ads,
                AVG(duration_seconds) as avg_duration,
                SUM(appearance_count) as total_appearances,
                SUM(CASE WHEN DATE(start_time) = CURDATE() THEN 1 ELSE 0 END) as today_occurrences,
                MAX(end_time) as last_event
            FROM lband_ad_events
            """
            
            cursor.execute(query)
            stats = cursor.fetchone()
            
            cursor.close()
            conn.close()
            return stats
            
        except Error as e:
            logger.error(f"[ERROR] Failed to fetch statistics: {e}")
            if conn:
                conn.close()
            return {}
    
    def cleanup_old_events(self, days=30):
        conn = self.get_connection()
        if not conn:
            return 0
        
        try:
            cursor = conn.cursor()
            
            query = """
            DELETE FROM lband_ad_events 
            WHERE end_time < DATE_SUB(NOW(), INTERVAL %s DAY)
            """
            
            cursor.execute(query, (days,))
            deleted = cursor.rowcount
            
            logger.info(f"[*] Cleaned up {deleted} old events")
            
            cursor.close()
            conn.close()
            return deleted
            
        except Error as e:
            logger.error(f"[ERROR] Failed to cleanup old events: {e}")
            if conn:
                conn.close()
            return 0
    
    def close_pool(self):
        try:
            if self.pool:
                logger.info("[OK] Database pool closed")
        except Exception as e:
            logger.error(f"[ERROR] Error closing pool: {e}")


# Singleton instance
_db_manager = None

def get_database_manager():
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("[*] Testing Database Module...")
    
    try:
        db = DatabaseManager()
        
        test_id = db.insert_ad_event(
            track_id=999,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=5.5,
            appearance_count=150
        )
        
        if test_id:
            print(f"[OK] Test insert successful: ID={test_id}")
        
        # Test statistics
        stats = db.get_statistics()
        print(f"[STATS] Statistics: {stats}")
        
        # Test recent events
        events = db.get_recent_events(5)
        print(f"[*] Recent events: {len(events)} found")
        
        print("[OK] All database tests passed!")
        
    except Exception as e:
        print(f"[ERROR] Database test failed: {e}")