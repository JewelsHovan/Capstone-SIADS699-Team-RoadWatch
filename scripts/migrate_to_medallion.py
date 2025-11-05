#!/usr/bin/env python3
"""
Migrate Data Structure to Medallion Architecture (Bronze/Silver/Gold)

This script reorganizes the data directory from:
  data/raw, data/processed, data/ml_ready
To:
  data/bronze, data/silver, data/gold

Bronze: Raw, immutable data (as downloaded)
Silver: Cleaned, validated, standardized
Gold: Business-level aggregates, ML-ready datasets

Usage:
    python scripts/migrate_to_medallion.py --dry-run   # Preview changes
    python scripts/migrate_to_medallion.py             # Execute migration
    python scripts/migrate_to_medallion.py --rollback  # Undo migration

Author: Data Engineering Team
Date: 2025-11-04
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import json

# Paths
DATA_DIR = Path('data')
BACKUP_DIR = Path('data_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

class MigrationPlan:
    """Define the migration plan"""

    def __init__(self):
        self.moves = []
        self.copies = []
        self.creates = []

    def add_move(self, source, dest, description=""):
        """Add a move operation"""
        self.moves.append({
            'source': Path(source),
            'dest': Path(dest),
            'description': description
        })

    def add_copy(self, source, dest, description=""):
        """Add a copy operation"""
        self.copies.append({
            'source': Path(source),
            'dest': Path(dest),
            'description': description
        })

    def add_create(self, path, description=""):
        """Add a directory creation"""
        self.creates.append({
            'path': Path(path),
            'description': description
        })

def create_migration_plan():
    """
    Create the migration plan

    Returns:
        MigrationPlan object
    """
    plan = MigrationPlan()

    # ============================================================================
    # 1. CREATE NEW DIRECTORY STRUCTURE
    # ============================================================================

    plan.add_create('data/bronze', 'Bronze layer (raw data)')
    plan.add_create('data/silver', 'Silver layer (cleaned data)')
    plan.add_create('data/gold', 'Gold layer (ML-ready data)')
    plan.add_create('data/gold/ml_datasets', 'ML training datasets')
    plan.add_create('data/gold/analytics', 'Analytics aggregates (future)')

    # ============================================================================
    # 2. BRONZE LAYER - Move raw data
    # ============================================================================

    # Move entire raw directory to bronze
    if Path('data/raw').exists():
        plan.add_move(
            'data/raw',
            'data/bronze',
            'Raw data (as downloaded, immutable)'
        )

    # ============================================================================
    # 3. SILVER LAYER - Move cleaned intermediate data
    # ============================================================================

    # Move processed HPMS from bronze to silver (it's actually cleaned, not raw)
    # Note: This will happen after raw→bronze move
    plan.add_move(
        'data/bronze/texas/roadway_characteristics',
        'data/silver/texas/roadway',
        'Cleaned HPMS road segments (extracted from GDB)'
    )

    # ============================================================================
    # 4. GOLD LAYER - Move ML-ready datasets
    # ============================================================================

    # Move ml_ready to gold/ml_datasets
    if Path('data/ml_ready/crash_level').exists():
        plan.add_move(
            'data/ml_ready/crash_level',
            'data/gold/ml_datasets/crash_level',
            'Crash-level ML datasets (train/val/test)'
        )

    if Path('data/ml_ready/segment_level').exists():
        plan.add_move(
            'data/ml_ready/segment_level',
            'data/gold/ml_datasets/segment_level',
            'Segment-level ML datasets (train/val/test)'
        )

    # Also move processed/crash_level to gold (has the timestamped versions)
    if Path('data/processed/crash_level').exists():
        plan.add_copy(
            'data/processed/crash_level',
            'data/gold/ml_datasets/crash_level_archive',
            'Archive of all crash-level dataset versions'
        )

    return plan

def print_plan(plan):
    """Print the migration plan"""
    print("\n" + "="*80)
    print("MIGRATION PLAN: Raw/Processed/ML_Ready → Bronze/Silver/Gold")
    print("="*80)

    # Directory creations
    if plan.creates:
        print("\n📁 DIRECTORIES TO CREATE:")
        for item in plan.creates:
            print(f"  ✓ {item['path']}")
            if item['description']:
                print(f"    └─ {item['description']}")

    # Moves
    if plan.moves:
        print("\n📦 FILES/FOLDERS TO MOVE:")
        for item in plan.moves:
            exists = "✓" if item['source'].exists() else "✗"
            size = get_dir_size(item['source']) if item['source'].exists() else 0
            print(f"  {exists} {item['source']}")
            print(f"    → {item['dest']}")
            if size > 0:
                print(f"    └─ Size: {format_size(size)}")
            if item['description']:
                print(f"    └─ {item['description']}")

    # Copies
    if plan.copies:
        print("\n📋 FILES/FOLDERS TO COPY:")
        for item in plan.copies:
            exists = "✓" if item['source'].exists() else "✗"
            size = get_dir_size(item['source']) if item['source'].exists() else 0
            print(f"  {exists} {item['source']}")
            print(f"    → {item['dest']}")
            if size > 0:
                print(f"    └─ Size: {format_size(size)}")
            if item['description']:
                print(f"    └─ {item['description']}")

    # Summary
    total_moves = len([m for m in plan.moves if m['source'].exists()])
    total_copies = len([c for c in plan.copies if c['source'].exists()])

    print("\n" + "="*80)
    print(f"SUMMARY: {len(plan.creates)} directories, {total_moves} moves, {total_copies} copies")
    print("="*80)

def get_dir_size(path):
    """Get total size of directory"""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size

    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except:
        pass
    return total

def format_size(bytes):
    """Format bytes as human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

def execute_migration(plan, dry_run=False):
    """
    Execute the migration plan

    Args:
        plan: MigrationPlan object
        dry_run: If True, only show what would be done
    """
    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")
        print_plan(plan)
        return True

    print("\n🚀 EXECUTING MIGRATION...")

    # Create backup
    print(f"\n📦 Creating backup: {BACKUP_DIR}/")
    if DATA_DIR.exists():
        try:
            shutil.copytree(DATA_DIR, BACKUP_DIR, symlinks=True)
            print(f"  ✓ Backup created successfully")
        except Exception as e:
            print(f"  ✗ Backup failed: {e}")
            print("  ⚠️  Migration aborted for safety")
            return False

    # Create directories
    print("\n📁 Creating directories...")
    for item in plan.creates:
        try:
            item['path'].mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {item['path']}")
        except Exception as e:
            print(f"  ✗ Failed to create {item['path']}: {e}")

    # Execute moves
    print("\n📦 Moving files/folders...")
    for item in plan.moves:
        if not item['source'].exists():
            print(f"  ⏭️  Skipped (not found): {item['source']}")
            continue

        try:
            # Create parent directory
            item['dest'].parent.mkdir(parents=True, exist_ok=True)

            # Move
            shutil.move(str(item['source']), str(item['dest']))
            print(f"  ✓ Moved: {item['source']}")
            print(f"     → {item['dest']}")
        except Exception as e:
            print(f"  ✗ Failed to move {item['source']}: {e}")

    # Execute copies
    print("\n📋 Copying files/folders...")
    for item in plan.copies:
        if not item['source'].exists():
            print(f"  ⏭️  Skipped (not found): {item['source']}")
            continue

        try:
            # Create parent directory
            item['dest'].parent.mkdir(parents=True, exist_ok=True)

            # Copy
            if item['source'].is_dir():
                shutil.copytree(str(item['source']), str(item['dest']),
                              symlinks=True, dirs_exist_ok=True)
            else:
                shutil.copy2(str(item['source']), str(item['dest']))

            print(f"  ✓ Copied: {item['source']}")
            print(f"     → {item['dest']}")
        except Exception as e:
            print(f"  ✗ Failed to copy {item['source']}: {e}")

    # Clean up old directories
    print("\n🧹 Cleaning up old structure...")
    old_dirs = ['data/processed', 'data/ml_ready']
    for old_dir in old_dirs:
        old_path = Path(old_dir)
        if old_path.exists():
            try:
                # Only remove if empty or just has old files
                if not any(old_path.iterdir()):
                    old_path.rmdir()
                    print(f"  ✓ Removed empty: {old_dir}")
                else:
                    print(f"  ⚠️  Kept (not empty): {old_dir}")
            except Exception as e:
                print(f"  ⚠️  Could not remove {old_dir}: {e}")

    # Save migration log
    log_file = Path('data/MIGRATION_LOG.json')
    try:
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'backup_location': str(BACKUP_DIR),
            'migrations': {
                'creates': [str(c['path']) for c in plan.creates],
                'moves': [{
                    'from': str(m['source']),
                    'to': str(m['dest'])
                } for m in plan.moves],
                'copies': [{
                    'from': str(c['source']),
                    'to': str(c['dest'])
                } for c in plan.copies]
            }
        }
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"\n📝 Migration log saved: {log_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save migration log: {e}")

    print("\n" + "="*80)
    print("✅ MIGRATION COMPLETE!")
    print("="*80)
    print(f"\n📦 Backup location: {BACKUP_DIR}/")
    print("\nNew structure:")
    print("  data/bronze/   - Raw, immutable data")
    print("  data/silver/   - Cleaned, validated data")
    print("  data/gold/     - ML-ready datasets")

    return True

def rollback_migration():
    """Rollback to backup"""
    print("\n🔄 ROLLBACK MODE")
    print("="*80)

    # Find most recent backup
    backups = sorted(Path('.').glob('data_backup_*'))

    if not backups:
        print("❌ No backups found!")
        return False

    latest_backup = backups[-1]
    print(f"\nLatest backup found: {latest_backup}")
    print(f"Created: {datetime.fromtimestamp(latest_backup.stat().st_mtime)}")

    response = input("\n⚠️  This will restore data from backup. Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Rollback cancelled")
        return False

    try:
        # Remove current data directory
        if DATA_DIR.exists():
            print(f"\n🗑️  Removing current: {DATA_DIR}")
            shutil.rmtree(DATA_DIR)

        # Restore from backup
        print(f"📦 Restoring from: {latest_backup}")
        shutil.copytree(latest_backup, DATA_DIR, symlinks=True)

        print("\n✅ Rollback complete!")
        return True

    except Exception as e:
        print(f"\n❌ Rollback failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Migrate data structure to medallion architecture (Bronze/Silver/Gold)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be changed
  python scripts/migrate_to_medallion.py --dry-run

  # Execute migration
  python scripts/migrate_to_medallion.py

  # Rollback to previous state
  python scripts/migrate_to_medallion.py --rollback

Architecture:
  Bronze: Raw, immutable data (as downloaded)
  Silver: Cleaned, validated, standardized
  Gold: Business-level aggregates, ML-ready datasets
        """
    )

    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without executing')
    parser.add_argument('--rollback', action='store_true',
                       help='Rollback to previous data structure')

    args = parser.parse_args()

    # Header
    print("\n" + "="*80)
    print("DATA MIGRATION: Medallion Architecture (Bronze/Silver/Gold)")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.rollback:
        return 0 if rollback_migration() else 1

    # Create migration plan
    plan = create_migration_plan()

    # Execute or preview
    if args.dry_run:
        print_plan(plan)
        print("\n💡 Run without --dry-run to execute migration")
        return 0
    else:
        # Show plan and confirm
        print_plan(plan)

        print("\n" + "="*80)
        print("⚠️  WARNING: This will reorganize your data directory")
        print("="*80)
        print("A backup will be created before making any changes.")
        response = input("\nProceed with migration? [y/N]: ")

        if response.lower() != 'y':
            print("\nMigration cancelled")
            return 0

        success = execute_migration(plan, dry_run=False)
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
