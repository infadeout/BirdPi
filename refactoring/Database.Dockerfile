FROM alpine:3.14

# Install SQLite and bash
RUN apk add --no-cache sqlite bash

WORKDIR /data

# Copy initialization script 
COPY ./database/init.sql /docker-entrypoint-initdb.d/
COPY ./database/entrypoint.sh /

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

# Create database directory with proper permissions
RUN mkdir -p /data && \
    adduser -D dbuser && \
    chown -R dbuser:dbuser /data && \
    chmod 777 /data

# Switch to non-root user
USER dbuser

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD sqlite3 /data/birds.db "SELECT 1" || exit 1

# Use entrypoint script to initialize database
ENTRYPOINT ["/entrypoint.sh"]

# Keep container running
CMD ["tail", "-f", "/dev/null"]