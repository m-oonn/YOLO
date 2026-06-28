#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Database migration script to add missing columns to events table."""

import os
import sqlite3
import sys

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

DB_PATH = os.path.join(COURSE_DIR, "outputs", "events.db")


def migrate_database():
    """Add missing columns to events table."""
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        print(
            "No migration needed - database will be created with correct schema on first run."
        )
        return True

    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Check current schema
    cur.execute("PRAGMA table_info(events)")
    columns = {row[1] for row in cur.fetchall()}
    print(f"\nCurrent columns: {', '.join(sorted(columns))}")

    # Columns to add
    migrations = [
        ("keypoints_json", "TEXT"),
        ("skeleton_count", "INTEGER DEFAULT 0"),
        ("priority", "TEXT DEFAULT 'INFO'"),
    ]

    changes_made = False
    for col_name, col_type in migrations:
        if col_name not in columns:
            print(f"\nAdding column: {col_name} {col_type}")
            try:
                cur.execute(f"ALTER TABLE events ADD COLUMN {col_name} {col_type}")
                print(f"  ✓ Column '{col_name}' added successfully")
                changes_made = True
            except sqlite3.OperationalError as e:
                print(f"  ✗ Failed to add column '{col_name}': {e}")
        else:
            print(f"\nColumn '{col_name}' already exists - skipping")

    if changes_made:
        conn.commit()
        print("\n✓ Migration completed successfully!")

        # Verify new schema
        cur.execute("PRAGMA table_info(events)")
        new_columns = {row[1] for row in cur.fetchall()}
        print(f"New columns: {', '.join(sorted(new_columns))}")
    else:
        print("\n✓ No migrations needed - database schema is up to date")

    conn.close()
    return True


if __name__ == "__main__":
    try:
        migrate_database()
        print("\nMigration script finished.")
    except Exception as e:
        print(f"\nMigration failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
