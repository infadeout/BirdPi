ROM alpine:3.14

# Install SQLite
RUN apk add --no-cache sqlite

WORKDIR /data

# Create initial database
COPY ./database/init.sql /docker-entrypoint-initdb.d/

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD sqlite3 /data/birds.db "SELECT 1" || exit 1

CMD ["sqlite3", "/data/birds.db"]