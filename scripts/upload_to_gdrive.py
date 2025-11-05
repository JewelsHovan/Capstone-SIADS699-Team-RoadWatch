#!/usr/bin/env python3
"""
Upload ML datasets to Google Drive

This script uploads the latest crash-level and segment-level datasets
to a specific Google Drive folder.

Setup:
    1. Install required packages:
       pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

    2. Set up Google Drive API:
       a. Go to https://console.cloud.google.com/
       b. Create a new project or select existing
       c. Enable "Google Drive API"
       d. Create OAuth 2.0 credentials (Desktop app)
       e. Download credentials.json to project root

    3. First run will open browser for authentication
       - Credentials are saved to token.json for future use

Usage:
    python upload_to_gdrive.py
    python upload_to_gdrive.py --folder-id YOUR_FOLDER_ID
    python upload_to_gdrive.py --crash-level-only
    python upload_to_gdrive.py --segment-level-only
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
except ImportError:
    print("❌ Error: Required packages not installed")
    print("\nInstall with:")
    print("  pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    sys.exit(1)

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Default folder ID (from user's shared link)
DEFAULT_FOLDER_ID = '1xVGXbxUFHSdSawo2C9wnmABj15wPEX3A'

# Paths
TOKEN_FILE = 'token.json'
CREDENTIALS_FILE = 'credentials.json'

# Use medallion architecture paths (Gold layer for ML datasets, Bronze for raw)
CRASH_LEVEL_DIR = Path('data/gold/ml_datasets/crash_level')
SEGMENT_LEVEL_DIR = Path('data/gold/ml_datasets/segment_level')
BRONZE_TEXAS_DIR = Path('data/bronze/texas')

def authenticate():
    """
    Authenticate with Google Drive API

    Returns:
        service: Google Drive API service object
    """
    creds = None

    # Load existing token if available
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"❌ Error: {CREDENTIALS_FILE} not found")
                print("\nPlease set up Google Drive API credentials:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create/select project")
                print("3. Enable Google Drive API")
                print("4. Create OAuth 2.0 credentials (Desktop app)")
                print(f"5. Download as {CREDENTIALS_FILE}")
                sys.exit(1)

            print("🔐 Authenticating with Google Drive...")
            print("   (Browser will open for authorization)")
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save credentials for future use
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
        print("✅ Credentials saved")

    # Build service
    service = build('drive', 'v3', credentials=creds)
    return service

def get_or_create_subfolder(service, parent_folder_id, subfolder_name, verbose=True):
    """
    Get existing subfolder or create if it doesn't exist

    Args:
        service: Google Drive service
        parent_folder_id: Parent folder ID
        subfolder_name: Name of subfolder
        verbose: Print progress

    Returns:
        Folder ID
    """
    # Search for existing folder
    query = f"name='{subfolder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"

    try:
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()

        items = results.get('files', [])

        if items:
            folder_id = items[0]['id']
            if verbose:
                print(f"  📁 Using existing folder: {subfolder_name}")
            return folder_id
        else:
            # Create folder
            file_metadata = {
                'name': subfolder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_folder_id]
            }
            folder = service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            folder_id = folder.get('id')
            if verbose:
                print(f"  📁 Created folder: {subfolder_name}")
            return folder_id

    except HttpError as error:
        print(f"❌ Error accessing folder: {error}")
        return None

def file_exists_in_folder(service, folder_id, filename):
    """
    Check if file already exists in folder

    Returns:
        File ID if exists, None otherwise
    """
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"

    try:
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()

        items = results.get('files', [])
        if items:
            return items[0]['id']
        return None

    except HttpError as error:
        print(f"❌ Error checking file: {error}")
        return None

def upload_file(service, file_path, folder_id, upload_name=None, replace=True, verbose=True):
    """
    Upload a file to Google Drive

    Args:
        service: Google Drive service
        file_path: Path to file to upload
        folder_id: Destination folder ID
        upload_name: Optional custom name for uploaded file (default: use original filename)
        replace: Replace if file exists
        verbose: Print progress

    Returns:
        File ID if successful, None otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"  ❌ File not found: {file_path}")
        return None

    file_size_mb = file_path.stat().st_size / 1024 / 1024

    # Use custom name if provided, otherwise use original filename
    final_name = upload_name if upload_name else file_path.name

    if verbose:
        if upload_name:
            print(f"\n  📤 Uploading: {file_path.name} → {final_name} ({file_size_mb:.1f} MB)")
        else:
            print(f"\n  📤 Uploading: {file_path.name} ({file_size_mb:.1f} MB)")

    try:
        # Check if file exists (using final name)
        existing_file_id = file_exists_in_folder(service, folder_id, final_name)

        if existing_file_id and not replace:
            print(f"  ⏭️  File already exists, skipping")
            return existing_file_id

        # Prepare file metadata
        file_metadata = {
            'name': final_name,
            'parents': [folder_id]
        }

        # Determine MIME type
        if file_path.suffix == '.csv':
            mime_type = 'text/csv'
        else:
            mime_type = 'application/octet-stream'

        media = MediaFileUpload(
            str(file_path),
            mimetype=mime_type,
            resumable=True
        )

        if existing_file_id:
            # Update existing file
            if verbose:
                print(f"  🔄 Replacing existing file...")
            file = service.files().update(
                fileId=existing_file_id,
                media_body=media
            ).execute()
        else:
            # Create new file
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

        file_id = file.get('id')

        if verbose:
            print(f"  ✅ Uploaded successfully")
            print(f"     File ID: {file_id}")
            print(f"     Link: https://drive.google.com/file/d/{file_id}/view")

        return file_id

    except HttpError as error:
        print(f"  ❌ Upload failed: {error}")
        return None

def upload_dataset_directory(service, dataset_dir, parent_folder_id, dataset_name, verbose=True):
    """
    Upload only latest train/val/test files from dataset directory with clean names

    Args:
        service: Google Drive service
        dataset_dir: Local directory path
        parent_folder_id: Google Drive parent folder ID
        dataset_name: Name for subfolder (e.g., 'crash_level')
        verbose: Print progress

    Returns:
        Number of files uploaded
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        print(f"❌ Directory not found: {dataset_dir}")
        return 0

    print(f"\n{'='*70}")
    print(f"📦 Uploading {dataset_name.upper()} Dataset")
    print(f"{'='*70}")

    # Create/get subfolder
    subfolder_id = get_or_create_subfolder(service, parent_folder_id, dataset_name, verbose)

    if not subfolder_id:
        print(f"❌ Failed to create/access subfolder")
        return 0

    # Define which files to upload and their clean names
    if dataset_name == 'crash_level':
        files_to_upload = [
            ('train_latest.csv', 'train.csv'),
            ('val_latest.csv', 'val.csv'),
            ('test_latest.csv', 'test.csv'),
            ('DATA_DICTIONARY.md', 'DATA_DICTIONARY.md'),
        ]
    elif dataset_name == 'segment_level':
        files_to_upload = [
            ('train_latest.csv', 'train.csv'),
            ('val_latest.csv', 'val.csv'),
            ('test_latest.csv', 'test.csv'),
            ('DATA_DICTIONARY.md', 'DATA_DICTIONARY.md'),
        ]
    else:
        # Fallback: upload all CSV files with original names
        csv_files = [f for f in dataset_dir.glob('*.csv') if not f.is_symlink()]
        files_to_upload = [(f.name, f.name) for f in csv_files]

    print(f"\n  Uploading {len(files_to_upload)} files (latest versions only)")

    # Upload each file
    uploaded = 0
    for local_name, upload_name in files_to_upload:
        file_path = dataset_dir / local_name

        # Resolve symlink to actual file
        if file_path.is_symlink():
            file_path = file_path.resolve()

        if not file_path.exists():
            if verbose:
                print(f"\n  ⚠️  File not found: {local_name}")
            continue

        file_id = upload_file(service, file_path, subfolder_id,
                            upload_name=upload_name, replace=True, verbose=verbose)
        if file_id:
            uploaded += 1

    print(f"\n✅ Uploaded {uploaded} files to {dataset_name}/")

    return uploaded

def upload_raw_texas_data(service, parent_folder_id, verbose=True):
    """
    Upload raw Texas datasets with clean names

    Args:
        service: Google Drive service
        parent_folder_id: Google Drive parent folder ID
        verbose: Print progress

    Returns:
        Number of files uploaded
    """

    if not BRONZE_TEXAS_DIR.exists():
        print(f"❌ Bronze Texas directory not found: {BRONZE_TEXAS_DIR}")
        return 0

    print(f"\n{'='*70}")
    print(f"📦 Uploading RAW TEXAS DATA")
    print(f"{'='*70}")

    # Create/get raw_texas subfolder
    raw_texas_folder_id = get_or_create_subfolder(service, parent_folder_id, 'raw_texas', verbose)

    if not raw_texas_folder_id:
        print(f"❌ Failed to create/access raw_texas folder")
        return 0

    # First upload the main data dictionary
    main_dict_path = BRONZE_TEXAS_DIR / 'DATA_DICTIONARY.md'
    if main_dict_path.exists():
        upload_file(service, main_dict_path, raw_texas_folder_id,
                   upload_name='DATA_DICTIONARY.md', replace=True, verbose=verbose)

    # Define datasets to upload from each subdirectory
    datasets = {
        'crashes': [
            ('crashes/kaggle_us_accidents_texas.csv', 'us_accidents_texas.csv'),
            ('crashes/austin_crashes_20251025_184712.csv', 'austin_crashes.csv'),
        ],
        'traffic': [
            ('traffic/txdot_aadt_annual.gpkg', 'txdot_aadt_annual.gpkg'),
        ],
        'workzones': [
            ('workzones/texas_wzdx_feed.csv', 'workzones.csv'),
            ('workzones/texas_wzdx_feed.json', 'workzones.json'),
        ],
        'weather': [
            ('weather/texas_weather_latest.csv', 'weather.csv'),
        ]
    }

    total_uploaded = 0

    for category, files in datasets.items():
        # Create category subfolder
        category_folder_id = get_or_create_subfolder(service, raw_texas_folder_id, category, verbose)

        if not category_folder_id:
            print(f"  ⚠️  Failed to create {category} subfolder")
            continue

        print(f"\n  📂 Uploading {category.upper()} data...")

        for local_path, upload_name in files:
            file_path = BRONZE_TEXAS_DIR / local_path

            # Resolve symlink if needed
            if file_path.is_symlink():
                file_path = file_path.resolve()

            if not file_path.exists():
                if verbose:
                    print(f"    ⚠️  File not found: {local_path}")
                continue

            file_id = upload_file(service, file_path, category_folder_id,
                                upload_name=upload_name, replace=True, verbose=verbose)
            if file_id:
                total_uploaded += 1

    print(f"\n✅ Uploaded {total_uploaded} raw Texas files")

    return total_uploaded

def main():
    parser = argparse.ArgumentParser(
        description="Upload ML datasets to Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload both datasets to default folder
  python upload_to_gdrive.py

  # Upload to specific folder
  python upload_to_gdrive.py --folder-id YOUR_FOLDER_ID

  # Upload only crash-level data
  python upload_to_gdrive.py --crash-level-only

  # Upload only segment-level data
  python upload_to_gdrive.py --segment-level-only

Setup:
  1. pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
  2. Create OAuth credentials at https://console.cloud.google.com/
  3. Download credentials.json to project root
  4. Run script (browser will open for auth first time)
        """
    )

    parser.add_argument('--folder-id', type=str, default=DEFAULT_FOLDER_ID,
                       help='Google Drive folder ID (default: from shared link)')
    parser.add_argument('--crash-level-only', action='store_true',
                       help='Upload only crash-level dataset')
    parser.add_argument('--segment-level-only', action='store_true',
                       help='Upload only segment-level dataset')
    parser.add_argument('--raw-only', action='store_true',
                       help='Upload only raw Texas data')
    parser.add_argument('--no-raw', action='store_true',
                       help='Skip raw Texas data upload')
    parser.add_argument('--no-replace', action='store_true',
                       help='Skip files that already exist')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("\n" + "="*70)
        print("🚀 GOOGLE DRIVE UPLOADER FOR ML DATASETS")
        print("="*70)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Destination: https://drive.google.com/drive/folders/{args.folder_id}")

    # Authenticate
    try:
        service = authenticate()
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

    if verbose:
        print("\n✅ Successfully authenticated with Google Drive")

    # Upload datasets
    total_uploaded = 0

    # Upload processed datasets (unless raw-only mode)
    if not args.raw_only:
        if not args.segment_level_only:
            uploaded = upload_dataset_directory(
                service, CRASH_LEVEL_DIR, args.folder_id, 'crash_level', verbose
            )
            total_uploaded += uploaded

        if not args.crash_level_only:
            uploaded = upload_dataset_directory(
                service, SEGMENT_LEVEL_DIR, args.folder_id, 'segment_level', verbose
            )
            total_uploaded += uploaded

    # Upload raw Texas data (unless explicitly skipped)
    if not args.no_raw and not args.crash_level_only and not args.segment_level_only:
        uploaded = upload_raw_texas_data(service, args.folder_id, verbose)
        total_uploaded += uploaded
    elif args.raw_only:
        uploaded = upload_raw_texas_data(service, args.folder_id, verbose)
        total_uploaded += uploaded

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("📊 UPLOAD SUMMARY")
        print("="*70)
        print(f"\nTotal files uploaded: {total_uploaded}")
        print(f"Destination: https://drive.google.com/drive/folders/{args.folder_id}")
        print("\n✅ Upload complete!")
        print("\nFiles uploaded with clean names:")
        if not args.raw_only:
            print("  crash_level/")
            print("    - train.csv")
            print("    - val.csv")
            print("    - test.csv")
            print("  segment_level/")
            print("    - train.csv")
            print("    - val.csv")
            print("    - test.csv")
        if not args.no_raw or args.raw_only:
            print("  raw_texas/")
            print("    crashes/")
            print("      - us_accidents_texas.csv")
            print("      - austin_crashes.csv")
            print("    traffic/")
            print("      - txdot_aadt_annual.gpkg")
            print("    workzones/")
            print("      - workzones.csv")
            print("      - workzones.json")
            print("    weather/")
            print("      - weather.csv")

    return 0

if __name__ == "__main__":
    sys.exit(main())
