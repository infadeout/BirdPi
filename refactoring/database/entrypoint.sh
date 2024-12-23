#!/bin/bash
set -e

# Initialize database if it doesn't exist
if [ ! -f /data/birds.db ]; then
    echo "Initializing database..."
    sqlite3 /data/birds.db < /docker-entrypoint-initdb.d/init.sql
    echo "Database initialized successfully"
fi

# Execute CMD
exec "$@"