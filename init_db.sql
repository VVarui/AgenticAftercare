-- AI Aftercare database
--
-- The schema is now created by the API itself at startup (db.py, Store.init_schema), so the
-- tables and the Python code cannot drift apart. This file is only mounted by docker compose
-- and intentionally contains no tables.
--
-- If you ran the OLD init_db.sql before (tables with a SERIAL "id" column), reset the empty
-- database once:  docker compose down ; docker volume rm <project>_postgres_data ; docker compose up -d
-- (find the volume name with: docker volume ls). Do not use 'down -v': it also deletes the Ollama models.
SELECT 1;
